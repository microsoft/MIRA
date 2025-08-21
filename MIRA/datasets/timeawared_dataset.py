#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os 
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter
import pickle 
import gzip
import yaml 
from time_moe.utils.log_util import logger
from time_moe.datasets.ts_dataset import TimeSeriesDataset
import traceback
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def quantize_time(times, 
                  initial_resolution=1.0, 
                  min_resolution=1e-8, 
                  shrink_factor=10, 
                  jitter_eps=1e-8, 
                  max_iterations=20):
    """
    Quantize time points while ensuring uniqueness by automatically adjusting resolution.
    If maximum iterations reached, resolve duplicates manually.

    Args:
        times (array-like): Original timestamps.
        initial_resolution (float): Starting quantization resolution.
        min_resolution (float): Minimum resolution limit.
        shrink_factor (int): Factor to shrink resolution per iteration.
        jitter_eps (float): Minimum additive noise to enforce strictly increasing times.
        max_iterations (int): Max shrink attempts.

    Returns:
        np.ndarray: Quantized timestamps in float32, guaranteed unique.
    """
    times = np.array(times, dtype=np.float64)
    resolution = initial_resolution

    for it in range(max_iterations):
        quantized = np.round(times / resolution) * resolution

        # Check if mapping is unique (more strict)
        counts = Counter(quantized)
        duplicates = [v for v, cnt in counts.items() if cnt > 1]

        if len(np.unique(quantized)) == len(times):
            print(f"[Info] Quantization succeeded at resolution {resolution:.8f} after {it+1} iterations.")
            return quantized.astype(np.float32)
        
        resolution = max(resolution / shrink_factor, min_resolution)

    # Fallback: Force uniqueness with dynamic jitter
    print(f"[Warning] Maximum iterations reached. Forcing uniqueness at resolution {resolution:.8f}.")
    quantized = np.round(times / resolution) * resolution
    unique_quantized = []
    last_value = None
    
    # Dynamically compute effective jitter
    max_abs_time = np.max(np.abs(times)) if len(times) > 0 else 1.0
    current_eps = max(jitter_eps, np.finfo(np.float32).eps * max_abs_time)
    current_eps = min(current_eps, max_abs_time * 1e-6)  # Prevent overflow

    for q in quantized:
        if last_value is not None and q <= last_value:
            q = last_value + current_eps
        unique_quantized.append(q)
        last_value = q

    # Convert to float32 and perform second validation
    times_value = np.array(unique_quantized, dtype=np.float32)
    if len(np.unique(times_value)) < len(times_value):
        # Handle rare duplicate cases
        _, indices = np.unique(times_value, return_index=True)
        duplicates = np.setdiff1d(np.arange(len(times_value)), indices)
        for idx in duplicates:
            times_value[idx] += current_eps

    return times_value

def read_file_by_extension(fn):
    if fn.endswith('.json'):
        with open(fn, encoding='utf-8') as file:
            data = json.load(file)
    elif fn.endswith('.jsonl'):
        data = read_jsonl_to_list(fn)
    elif fn.endswith('.yaml'):
        data = load_yaml_file(fn)
    elif fn.endswith('.npy'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npz'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npy.gz'):
        with gzip.GzipFile(fn, 'r') as file:
            data = np.load(file, allow_pickle=True)
    elif fn.endswith('.pkl') or fn.endswith('.pickle'):
        data = load_pkl_obj(fn)
    else:
        raise RuntimeError(f'Unknown file extension: {fn}')
    return data

def read_jsonl_to_list(jsonl_fn):
    with open(jsonl_fn, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file.readlines()]

def load_yaml_file(fn):
    with open(fn, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_pkl_obj(fn):
    out_list = []
    with open(fn, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                out_list.append(data)
            except EOFError:
                break
    if len(out_list) == 0:
        return None
    elif len(out_list) == 1:
        return out_list[0]
    else:
        return out_list

class TimeAwareJSONLDataset(TimeSeriesDataset):
    # Keep __init__ largely the same, but add time normalization fitting
    def __init__(self, data_path, time_normalization='standard', quantize_resolution=None, auto_quantize=False, sample_size=1000, data_normalizer = MinMaxScaler()):
        """
        Args:
            data_path (str): Path to the .jsonl file.
            time_normalization (str or None): 'standard' for standardization, None to disable.
            quantize_resolution (float, optional): Time quantization resolution. Defaults to None.
            auto_quantize (bool, optional): Infer quantization resolution. Defaults to False.
            sample_size (int, optional): Number of samples to use for inferring resolution/normalization. Defaults to 1000.
        """
        if not os.path.exists(data_path) or not data_path.endswith('.jsonl'):
             raise ValueError(f"Invalid data path: {data_path}. Expecting a .jsonl file.")

        logger.info(f"Loading data from {data_path}...")
        self.data = read_file_by_extension(data_path)
        self.num_tokens = None
        self.quantize_resolution = quantize_resolution
        self.time_normalizer = None
        self.data_normalizer = data_normalizer
        # Fit data normalizer on all sequences
        self._fit_data_normalizer()
        # Fit time normalizer and infer quantization

        logger.info(f"Finished loading and preprocessing meta info for {data_path}.")

    def _fit_data_normalizer(self):
        all_vals = []
        for item in self.data:
            seq = None
            if isinstance(item, dict) and 'sequence' in item:
                seq = np.asarray(item['sequence'], dtype=np.float64)
            elif isinstance(item, (list, np.ndarray)):
                seq = np.asarray(item, dtype=np.float64)
            if seq is None or seq.ndim != 1 or len(seq) == 0:
                continue
            all_vals.append(seq.reshape(-1, 1))
        if not all_vals:
            logger.warning("No valid sequence data available. Disabling data normalization.")
            self.data_normalizer = None
            return
        all_data = np.vstack(all_vals)
        try:
            self.data_normalizer.fit(all_data)
            logger.info(f"Fitted data normalizer {self.data_normalizer} on {all_data.shape[0]} values.")
        except Exception as e:
            logger.error(f"Error fitting data normalizer: {e}. Disabling normalization.")
            self.data_normalizer = None


    def _fit_normalizer_and_infer_quantization(self, time_normalization, auto_quantize, sample_size):
        """Fit time normalizer and optionally infer quantization resolution."""
        all_times = []
        all_deltas = []
        num_items_to_sample = min(sample_size, len(self.data))
        logger.info(f"Sampling {num_items_to_sample} items for normalization/quantization info...")

        # Sample items efficiently if dataset is large
        indices_to_sample = np.random.choice(len(self.data), num_items_to_sample, replace=False)

        for i in indices_to_sample:
            item = self.data[i]
            if isinstance(item, dict) and 'time' in item and 'sequence' in item:
                # Ensure mask exists or default to all valid
                mask = np.array(item.get('mask', np.ones_like(item['sequence'])), dtype=int)
                time = np.array(item['time'], dtype=np.float64) # Use float64 for stats
                sequence = np.array(item['sequence'])

                # Check consistency
                if len(time) != len(sequence) or len(mask) != len(sequence):
                    logger.warning(f"Inconsistent lengths in item {i}: seq={len(sequence)}, time={len(time)}, mask={len(mask)}. Skipping item.")
                    continue

                valid_times = time[mask == 1]
                if len(valid_times) > 0:
                    all_times.append(valid_times)
                    if len(valid_times) >= 2:
                        deltas = np.diff(valid_times)
                        deltas = deltas[deltas > 0] # Only positive deltas
                        if len(deltas) > 0:
                            all_deltas.append(deltas)
            # Handle case where item is just a sequence (assuming regular time)
            elif isinstance(item, (list, np.ndarray)):
                 # Cannot infer time normalization or quantization without time info
                 pass


        if not all_times:
             logger.warning("No valid time data found in the sample to compute normalization statistics.")
             return

        all_times_flat = np.concatenate(all_times)


        # Fit Time Normalizer
        if time_normalization == 'standard':
            self.time_normalizer = StandardScaler()
            # ... (try/except for standard scaler) ...
        elif time_normalization == 'minmax': #
            self.time_normalizer = MinMaxScaler(feature_range=(0, 1)) 
            try:
                self.time_normalizer.fit(all_times_flat.reshape(-1, 1))

                data_min = getattr(self.time_normalizer, 'data_min_', [np.nan])[0]
                data_max = getattr(self.time_normalizer, 'data_max_', [np.nan])[0]
                scale = getattr(self.time_normalizer, 'scale_', [np.nan])[0]
                min_ = getattr(self.time_normalizer, 'min_', [np.nan])[0]
                logger.info(f"Fitted MinMaxScaler for time: data_min={data_min:.4f}, data_max={data_max:.4f}, scale={scale:.4f}, min_={min_:.4f}")
            except ValueError as e:
                logger.error(f"Error fitting MinMaxScaler: {e}. Disabling time normalization.")
                self.time_normalizer = None
        elif time_normalization is None or time_normalization.lower() == 'none':
            logger.info("Time normalization is disabled.")
            self.time_normalizer = None
        else: 
            logger.warning(f"Unsupported time_normalization_method: {time_normalization}. Normalization disabled.")
            self.time_normalizer = None


        # Infer Quantization Resolution (if enabled and not provided)
        if auto_quantize and self.quantize_resolution is None:
            if not all_deltas:
                logger.warning("[Warning] Cannot infer time resolution from deltas, default to 1.0")
                self.quantize_resolution = 1.0
            else:
                all_deltas_flat = np.concatenate(all_deltas)
                if len(all_deltas_flat) == 0:
                     logger.warning("[Warning] No positive time deltas found, default to 1.0")
                     self.quantize_resolution = 1.0
                else:
                     # Use median of positive deltas as estimated resolution
                     estimated_resolution = np.median(all_deltas_flat)
                     self.quantize_resolution = max(estimated_resolution, 1e-9) # Avoid zero resolution
                     logger.info(f"[Info] Inferred quantize resolution: {self.quantize_resolution:.6f}")
        elif self.quantize_resolution is not None:
             logger.info(f"Using provided quantize resolution: {self.quantize_resolution:.6f}")

    
    def get_sequence_length_by_idx(self, seq_idx):

        try:
            length = self.get_sequence_length(seq_idx)
            if not isinstance(length, int) or length < 0:
                 logger.error(f"Internal Error: get_sequence_length returned invalid value {length} for seq_idx {seq_idx}. Treating as 0.")
                 return 0
            return length
        except Exception as e: 
            logger.error(f"CRITICAL: Unexpected error in get_sequence_length_by_idx for seq_idx {seq_idx}: {type(e).__name__} - {e}.")
            logger.error(traceback.format_exc())
            return 0 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_idx):
        item = self.data[seq_idx]
        if isinstance(item, dict):
            # Assume keys 'sequence', 'time', 'mask' exist based on format
            sequence = np.array(item['sequence'], dtype=np.float32)
            time = np.array(item['time'], dtype=np.float64) # Load as float64
            # Default mask to all ones if missing
            mask = np.array(item.get('mask', np.ones_like(sequence)), dtype=int)

            # Validate lengths
            if not (len(sequence) == len(time) == len(mask)):
                raise ValueError(f"Data inconsistency at index {seq_idx}: lengths differ. "
                                 f"Sequence: {len(sequence)}, Time: {len(time)}, Mask: {len(mask)}")

        elif isinstance(item, (list, np.ndarray)): # Handle case where only sequence is provided
            sequence = np.array(item, dtype=np.float32)
            # Create dummy time and mask if missing
            time = np.arange(len(sequence), dtype=np.float64)
            mask = np.ones(len(sequence), dtype=int)
            logger.warning(f"Item at index {seq_idx} only contains sequence data. Creating default time and mask.", once=True)
        else:
            raise TypeError(f"Unsupported data type at index {seq_idx}: {type(item)}")

        # Apply quantization if specified
        if self.quantize_resolution is not None:
            time = quantize_time(time, initial_resolution=self.quantize_resolution) # Use float32 result

        if self.data_normalizer is not None:
            seq = np.asarray(item['sequence'], dtype=np.float32)
            seq = self.data_normalizer.transform(seq.reshape(-1,1)).reshape(-1)
        # Return data including mask and original times (normalization applied later)
        # Return time as float32 for consistency in tensors
        return {
            'sequence': seq,
            'time': time.astype(np.float32),
            'mask': mask.astype(np.int32) # Use int32 for mask
        }

    def get_num_tokens(self):
        if self.num_tokens is None:
            logger.info("Calculating total number of tokens...")
            self.num_tokens = sum(self.get_sequence_length(i) for i in range(len(self)))
        return self.num_tokens

    def get_sequence_length(self, seq_idx):
        try:
            item = self.data[seq_idx] 
    
            if isinstance(item, dict):
                sequence_data = item.get('sequence') 
                if sequence_data is not None and hasattr(sequence_data, '__len__'):
                    return len(sequence_data) 
                else: 
                    logger.warning(f"Item at seq_idx {seq_idx} is dict but 'sequence' key is problematic. Returning length 0.", once=True)
                    return 0
            elif item is not None and hasattr(item, '__len__'): 
                return len(item)
            else: 
                logger.warning(f"Item at seq_idx {seq_idx} is None or not sequence-like (type: {type(item)}). Returning length 0.", once=True)
                return 0
        except IndexError:
            logger.error(f"IndexError getting length for seq_idx {seq_idx}. Max index is {len(self.data)-1}. Returning 0.")
            return 0
        except Exception as e: 
            logger.error(f"Unexpected error getting sequence length for seq_idx {seq_idx}: {e}. Returning 0.")
            return 0 

    def get_time_normalizer(self):
        """Returns the fitted time normalizer."""
        return self.time_normalizer

class MinMaxScalerFeatureRange(MinMaxScaler):
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        super().__init__(feature_range=feature_range, copy=copy, clip=clip)
    
    @property
    def mean_(self): # Dummy for consistent logging if trying to access normalizer.mean_
        return [np.nan]

class TimeAwareEvalDataset(Dataset):
    def __init__(self, dataset, context_length, prediction_length, normalize=False):
        self.source_dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_length = context_length + prediction_length
        self.normalize = normalize

        # Precompute all valid windows
        self.valid_windows = []
        for seq_idx in range(len(dataset)):
            seq_len = dataset.get_sequence_length(seq_idx)
            if seq_len >= self.window_length:
                for start in range(seq_len - self.window_length + 1):
                    self.valid_windows.append((seq_idx, start))

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        seq_idx, start = self.valid_windows[idx]
        end = start + self.window_length
        item = self.source_dataset[seq_idx]

        # Retrieve full window data
        full_sequence = item['sequence'][start:end]
        full_mask = item['mask'][start:end]
        full_time = item['time'][start:end] if item['time'] is not None else None

        # Split into context and prediction parts
        context_seq = full_sequence[:self.context_length]
        context_mask = full_mask[:self.context_length]
        pred_seq = full_sequence[self.context_length:]
        pred_mask = full_mask[self.context_length:]

        # Keep only valid (observed) values
        inputs = self._get_valid_values(context_seq, context_mask, 'inputs')
        labels = self._get_valid_values(pred_seq, pred_mask, 'labels')

        # Process time information
        if full_time is not None:
            context_time = full_time[:self.context_length]
            pred_time = full_time[self.context_length:]
            inputs['time'] = context_time[context_mask == 1]
            labels['time'] = pred_time[pred_mask == 1]

        if self.normalize:
            inputs, labels = self._normalize(inputs, labels)

        return {
            'inputs': inputs,
            'labels': labels,
            # Keep original mask information for further processing
            'input_mask': context_mask,
            'label_mask': pred_mask
        }

    def _get_valid_values(self, sequence, mask, prefix):
        """Extract valid values and record their original indices."""
        valid_mask = mask == 1
        valid_indices = np.where(valid_mask)[0]
        return {
            'sequence': sequence[valid_mask],
            'valid_indices': valid_indices,
            'original_length': len(sequence)
        }

    def _normalize(self, inputs, labels):
        """Normalize based on valid values only."""
        if len(inputs['sequence']) > 0:
            mean = inputs['sequence'].mean()
            std = inputs['sequence'].std()
            std = 1.0 if std == 0 else std

            inputs['sequence'] = (inputs['sequence'] - mean) / std
            labels['sequence'] = (labels['sequence'] - mean) / std

        return inputs, labels