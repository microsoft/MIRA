# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from mira.runner import MIRARunner
import re
import os


"""
Our original code 
Checkpoint Detection Utility
"""

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

"""
Part of code from time_moe.main
https://github.com/Time-MoE/Time-MoE
"""

def get_last_checkpoint(folder):
    if not os.path.isdir(folder):
        return None
    content = os.listdir(folder)
    checkpoints = [
        path for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return None
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, help='Path to training data. (Folder contains data files, or data file)')
    parser.add_argument('--model_path', '-m', type=str, default='Maple728/TimeMoE-50M', help='Path to pretrained model. Default: Maple728/TimeMoE-50M')
    parser.add_argument('--output_path', '-o', type=str, default='logs/time_moe')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=0, help='minimum learning rate')

    parser.add_argument('--train_steps', type=int, default=None, help='number of training steps')
    parser.add_argument('--num_train_epochs', type=float, default=1.0, help='number of training epochs')
    parser.add_argument('--normalization_method', type=str, choices=['none', 'zero', 'max'], default='zero', help='normalization method for sequence')

    parser.add_argument('--seed', type=int, default=9899, help='random seed')
    parser.add_argument('--attn_implementation', type=str, choices=['auto', 'eager', 'flash_attention_2'], default='auto', help='attention implementation')
    
    parser.add_argument('--lr_scheduler_type', type=str, choices=['constant', 'linear', 'cosine', 'constant_with_warmup'], default='linear', help='learning rate scheduler type')
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help='warmup ratio')
    parser.add_argument('--warmup_steps', type=int, default=0, help='warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    
    parser.add_argument('--global_batch_size', type=int, default=16, help='global batch size')
    parser.add_argument('--micro_batch_size', type=int, default=16, help='micro batch size per device')
    
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], type=str, default='fp32', help='precision mode (default: fp32)')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='enable gradient checkpointing')
    parser.add_argument('--deepspeed', type=str, default=None, help='DeepSpeed config file path')

    parser.add_argument('--from_scratch', action='store_true', help='train from scratch')
    parser.add_argument('--save_steps', type=int, default=None, help='number of steps to save model')
    parser.add_argument('--save_strategy', choices=['steps', 'epoch', 'no'], type=str, default='no', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=None, help='limit the number of checkpoints')
    parser.add_argument('--save_only_model', action='store_true', help='save only model')

    parser.add_argument('--logging_steps', type=int, default=1, help='number of steps to log')
    parser.add_argument('--evaluation_strategy', choices=['steps', 'epoch', 'no'], type=str, default='no', help='evaluation strategy')
    parser.add_argument('--eval_steps', type=int, default=None, help='number of evaluation steps')

    parser.add_argument('--adam_beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.95, help='adam beta2')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max gradient norm')
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='number of workers for dataloader')

    # --- New Arguments for Time-Aware Dataset ---
    parser.add_argument('--time_aware_dataset', action='store_true', help='Use TimeAwareJSONLDataset and TimeAwareWindowDataset if data is .jsonl.')
    parser.add_argument('--time_normalization_method', type=str, choices=['none', 'standard', 'minmax'], default='none', help='Normalization method for timestamps (if using time_aware_dataset).')
    parser.add_argument('--time_quantize_resolution', type=float, default=None, help='Time quantization resolution (if using time_aware_dataset).')
    parser.add_argument('--time_auto_quantize', action='store_true', help='Automatically infer time quantization resolution (if using time_aware_dataset).')
    parser.add_argument('--data_sample_size', type=int, default=1000, help='Number of samples for inferring time normalization/quantization stats.')
    parser.add_argument('--min_valid_history', type=int, default=1, help='Minimum valid points required in a history window for TimeAwareWindowDataset.')
    parser.add_argument('--time_aware_rotary', action='store_true', help='Enable CT-RoPE positional encoding (default: disabled unless model supports it).')
    parser.add_argument('--time_scale', type=float, default=1.0, help='Scaling factor α for CT-RoPE time normalization (controls how time is mapped to RoPE index).')


    args = parser.parse_args()

    if args.normalization_method == 'none':
        args.normalization_method = None
    
    if args.time_normalization_method == 'none': # For timestamp normalization
        args.time_normalization_method = None
    
    last_checkpoint = get_last_checkpoint(args.output_path)
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("No interrupt checkpoint found. Starting training from scratch.")

    runner = MIRARunner(
        model_path=args.model_path,
        output_path=args.output_path,
        seed=args.seed,
    )

    runner.train_model(
        from_scratch=args.from_scratch,
        max_length=args.max_length,
        data_path=args.data_path,
        normalization_method=args.normalization_method,
        attn_implementation=args.attn_implementation,

        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        
        train_steps=args.train_steps,
        num_train_epochs=args.num_train_epochs,
        
        precision=args.precision,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        save_only_model=args.save_only_model,
        save_total_limit=args.save_total_limit,

        time_aware_dataset=args.time_aware_dataset,
        time_normalization_method=args.time_normalization_method,
        time_quantize_resolution=args.time_quantize_resolution,
        time_auto_quantize=args.time_auto_quantize,
        data_sample_size=args.data_sample_size,
        min_valid_history=args.min_valid_history,
        time_aware_rotary=args.time_aware_rotary,
        time_scale=args.time_scale,
        resume_from_checkpoint=last_checkpoint
    )
