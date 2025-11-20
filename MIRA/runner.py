'''
Code from time_moe.models.runner
https://github.com/Time-MoE
'''

import os
import math
import random
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler 

from mira.datasets.mira_dataset import MIRADataset
from mira.datasets.mira_window_dataset import MIRAWindowDataset, TimeAwareWindowDataset
from mira.models.modeling_mira import MIRAForPrediction, MIRAConfig
from mira.trainer.hf_trainer import MIRATrainingArguments, MIRATrainer
from mira.utils.dist_util import get_world_size
from mira.utils.log_util import logger, log_in_local_rank_0

from mira.datasets.timeawared_dataset import TimeAwareJSONLDataset

from mira.datasets.time_utils import time_aware_collate_fn


class MIRARunner:
    def __init__(
            self,
            model_path: str = None,
            output_path: str = 'logs/mira',
            seed: int = 9899
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.seed = seed

    def load_model(self, model_path: str = None, from_scatch: bool = False, **kwargs):
        if model_path is None:
            model_path = self.model_path

        attn = kwargs.pop('attn_implementation', None)
        if attn is None:
            attn = 'eager'
        elif attn == 'auto':
            # try to use flash-attention
            # Please aware mira cannot using falsh attention directly when using CT-RoPE
            try:
                from flash_attn import flash_attn_func, flash_attn_varlen_func
                from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
                attn = 'flash_attention_2'
            except:
                log_in_local_rank_0('Flash attention import failed, switching to eager attention.', type='warn')
                attn = 'eager'
        
        if attn == 'eager':
            log_in_local_rank_0('Use Eager Attention')
        elif attn == 'flash_attention_2':
            log_in_local_rank_0('Use Flash Attention 2')
        else:
            raise ValueError(f'Unknown attention method: {attn}')
        kwargs['attn_implementation'] = attn

        if from_scatch:
            config = MIRAConfig.from_pretrained(model_path, _attn_implementation=attn)
            model = MIRAForPrediction(config)
        else:
            model = MIRAForPrediction.from_pretrained(model_path, **kwargs)
        return model

    def train_model(self, from_scratch: bool = False, resume_from_checkpoint: str = None, **kwargs):
        setup_seed(self.seed)

        train_config = kwargs

        num_devices = get_world_size()
        
        global_batch_size = train_config.get('global_batch_size', None)
        micro_batch_size = train_config.get('micro_batch_size', None)

        if global_batch_size is None and micro_batch_size is None:
            raise ValueError('Must set at lease one argument: "global_batch_size" or "micro_batch_size"')
        elif global_batch_size is None:
            gradient_accumulation_steps = 1
            global_batch_size = micro_batch_size * num_devices
        elif micro_batch_size is None:
            micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = 1
        else:
            if micro_batch_size * num_devices > global_batch_size:
                if num_devices > global_batch_size:
                    micro_batch_size = 1
                    global_batch_size = num_devices
                else:
                    micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = math.ceil(global_batch_size / num_devices / micro_batch_size)
            global_batch_size = int(gradient_accumulation_steps * num_devices * micro_batch_size)

        if ('train_steps' in train_config
                and train_config['train_steps'] is not None
                and train_config['train_steps'] > 0):
            train_steps = int(train_config["train_steps"])
            num_train_epochs = -1
        else:
            train_steps = -1
            num_train_epochs = _safe_float(train_config.get("num_train_epochs", 1))

        precision = train_config.get('precision', 'bf16')
        if precision not in ['bf16', 'fp16', 'fp32']:
            log_in_local_rank_0(f'Precision {precision} is not set, use fp32 default!', type='warn')
            precision = 'fp32'

        if precision == 'bf16':
            torch_dtype = torch.bfloat16
        elif precision == 'fp16':
            # use fp32 to load model but uses fp15 to train model
            torch_dtype = torch.float32
        elif precision == 'fp32':
            torch_dtype = torch.float32
        else:
            raise ValueError(f'Unsupported precision {precision}')
        
        '''
        #### Our original code ####
        '''

        # --- Load Dataset ---
        is_time_aware_dataset = train_config.get('time_aware_dataset', True) # Default to True if using jsonl
        data_path = train_config['data_path']
        max_length = train_config['max_length'] # This is context_length

        current_data_collator = None # Default for non-time-aware

        if is_time_aware_dataset and data_path.endswith('.jsonl'):
            # 1. Instantiate the base JSONL dataset
            # Pass normalization choice and quantization options from config
            time_normalization = train_config.get('time_normalization', "none") # e.g., 'standard' or None
            quantize_res = train_config.get('time_quantize_resolution', None)
            auto_quantize = train_config.get('time_auto_quantize', False)
            sample_size = train_config.get('data_sample_size', 1000)

            base_dataset = TimeAwareJSONLDataset(
                data_path=data_path,
                time_normalization=time_normalization,
                quantize_resolution=quantize_res,
                auto_quantize=auto_quantize,
                sample_size=sample_size
            )

            window_dataset = TimeAwareWindowDataset(
                dataset=base_dataset,
                context_length=max_length,
                prediction_length=0, # For autoregressive training
                #time_normalizer=time_normalizer,
                min_valid_history=train_config.get('min_valid_history', 1) # Add config option
            )
            train_ds = window_dataset # Use this as the training dataset

        else:
             # Fallback to original non-time-aware data loading if needed
             log_in_local_rank_0('Loading non-time-aware dataset...')
             dataset = MIRADataset(
                 data_path,
                 normalization_method=train_config["normalization_method"] # Value normalization
             )
             log_in_local_rank_0('Processing dataset to fixed-size sub-sequences...')

             window_dataset = MIRAWindowDataset(
                 dataset,
                 context_length=max_length,
                 prediction_length=0,
                 shuffle=False
             )
             train_ds = window_dataset
        
        '''
        Code from time_moe.models.runner
        https://github.com/Time-MoE/Time-MoE
        '''

        log_in_local_rank_0(f'Set global_batch_size to {global_batch_size}')
        log_in_local_rank_0(f'Set micro_batch_size to {micro_batch_size}')
        log_in_local_rank_0(f'Set gradient_accumulation_steps to {gradient_accumulation_steps}')
        log_in_local_rank_0(f'Set precision to {precision}')
        log_in_local_rank_0(f'Set normalization to {train_config["normalization_method"]}')

        training_args = MIRATrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=num_train_epochs,
            # use_cpu=True,
            max_steps=train_steps,
            evaluation_strategy=train_config.get("evaluation_strategy", 'no'),
            eval_steps=_safe_float(train_config.get("eval_steps", None)),
            save_strategy=train_config.get("save_strategy", "no"),
            save_steps=_safe_float(train_config.get("save_steps", None)),
            learning_rate=float(train_config.get("learning_rate", 1e-5)),
            min_learning_rate=float(train_config.get("min_learning_rate", 0)),
            adam_beta1=float(train_config.get("adam_beta1", 0.9)),
            adam_beta2=float(train_config.get("adam_beta2", 0.95)),
            adam_epsilon=float(train_config.get("adam_epsilon", 1e-8)),
            lr_scheduler_type=train_config.get("lr_scheduler_type", 'constant'),
            warmup_ratio=float(train_config.get("warmup_ratio") or 0.0),
            warmup_steps=int(train_config.get("warmup_steps", 0)),
            weight_decay=float(train_config.get("weight_decay", 0.1)),
            per_device_train_batch_size=int(micro_batch_size),
            per_device_eval_batch_size=int(micro_batch_size * 2),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            gradient_checkpointing=train_config.get("gradient_checkpointing", False),
            bf16=True if precision == 'bf16' else False,
            fp16=True if precision == 'fp16' else False,
            deepspeed=train_config.get("deepspeed"),
            push_to_hub=False,
            logging_first_step=True,
            log_on_each_node=False,
            logging_steps=int(train_config.get('logging_steps', 1)),
            seed=self.seed,
            data_seed=self.seed,
            max_grad_norm=train_config.get('max_grad_norm', 1.0),
            optim=train_config.get('optim', 'adamw_torch'),
            torch_compile=train_config.get('torch_compile', False),
            dataloader_num_workers=train_config.get('dataloader_num_workers') or 2,
            ddp_find_unused_parameters=False,

            logging_dir=os.path.join(self.output_path, 'tb_logs'),
            save_only_model=train_config.get('save_only_model', True),
            save_total_limit=train_config.get('save_total_limit'),

            # data_collator=time_aware_collate_fn if isinstance(train_ds, TimeAwareWindowDataset) else None,
        )

        model_path = train_config.pop('model_path', None) or self.model_path
        if model_path is not None:
            model = self.load_model(
                model_path=model_path,
                from_scatch=from_scratch,
                torch_dtype=torch_dtype,
                attn_implementation=train_config.get('attn_implementation', 'eager'),
            )
            log_in_local_rank_0(f'Load model parameters from: {model_path}')
        else:
            raise ValueError('Model path is None')

        num_total_params = 0
        for p in model.parameters():
            num_total_params += reduce(mul, p.shape)

        # print statistics info
        log_in_local_rank_0(train_config)
        log_in_local_rank_0(training_args)
        log_in_local_rank_0(model.config)
        log_in_local_rank_0(f'Number of the model parameters: {length_to_str(num_total_params)}')

        if train_steps > 0:
            total_train_tokens = train_steps * global_batch_size * train_config['max_length']
            log_in_local_rank_0(f'Tokens will consume: {length_to_str(total_train_tokens)}')
   
        if train_ds is None:
             raise RuntimeError("train_ds was not properly initialized. Check dataset loading logic.")
        trainer = MIRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(self.output_path)
        log_in_local_rank_0(f'Saving model to {self.output_path}')

        return trainer.model

def setup_seed(seed: int = 9899):
    """
    Setup seed for all known operations.

    Args:
        seed (int): seed number.

    Returns:

    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def length_to_str(length):
    if length >= 1e12:
        return f'{length / 1e12:.3f}T'
    if length >= 1e9:
        return f'{length / 1e9:.3f}B'
    elif length >= 1e6:
        return f'{length / 1e6:.3f}M'
    else:
        return f'{length / 1e3:.3f}K'


def _safe_float(number):
    if number is None:
        return None
    else:
        return float(number)
