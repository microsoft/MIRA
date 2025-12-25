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
from mira.utils.log_util import logger
from mira.datasets.ts_dataset import TimeSeriesDataset
import traceback
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

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
    """
    A time-aware dataset loader for JSONL-based time series.
    Supports optional time normalization (standard/minmax/none),
    automatic quantization, and sequence normalization.
    """

    def __init__(
        self,
        data_path,
        time_normalization="none",     
        quantize_resolution=None,
        auto_quantize=False,
        sample_size=1000,
        data_normalizer=MinMaxScaler(),
    ):
        if not os.path.exists(data_path) or not data_path.endswith(".jsonl"):
            raise ValueError(f"Invalid data path: {data_path}. Expecting a .jsonl file.")

        logger.info(f"Loading data from {data_path}...")
        self.data = self._read_jsonl(data_path)
        self.num_tokens = None
        self.quantize_resolution = quantize_resolution
        self.auto_quantize = auto_quantize
        self.data_normalizer = data_normalizer
        self.time_normalization = time_normalization.lower() if isinstance(time_normalization, str) else None
        self.time_normalizer = None

        # Fit normalizers
        self._fit_data_normalizer()
        self._fit_time_normalizer(sample_size)

        logger.info(f"Finished loading and preprocessing meta info for {data_path}.")

    def _read_jsonl(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [eval(line.strip()) for line in f.readlines()]

    def _fit_data_normalizer(self):
        """Fit value normalizer (MinMaxScaler or StandardScaler) across all sequences."""
        all_vals = []
        for item in self.data:
            seq = None
            if isinstance(item, dict) and "sequence" in item:
                seq = np.asarray(item["sequence"], dtype=np.float64)
            elif isinstance(item, (list, np.ndarray)):
                seq = np.asarray(item, dtype=np.float64)
            if seq is None or seq.ndim != 1 or len(seq) == 0:
                continue
            all_vals.append(seq.reshape(-1, 1))

        if not all_vals:
            logger.warning("No valid sequence data found. Disabling value normalization.")
            self.data_normalizer = None
            return

        all_data = np.vstack(all_vals)
        try:
            self.data_normalizer.fit(all_data)
            logger.info(f"Fitted data normalizer on {all_data.shape[0]} values.")
        except Exception as e:
            logger.error(f"Error fitting data normalizer: {e}. Normalization disabled.")
            self.data_normalizer = None

    def _fit_time_normalizer(self, sample_size=1000):
        """Fit time normalization (optional) and infer quantization."""
        all_times, all_deltas = [], []
        sample_size = min(sample_size, len(self.data))
        indices = random.sample(range(len(self.data)), sample_size)

        for i in indices:
            item = self.data[i]
            if isinstance(item, dict) and "time" in item:
                time = np.asarray(item["time"], dtype=np.float64)
                mask = np.asarray(item.get("mask", np.ones_like(time)), dtype=int)
                valid = time[mask == 1]
                if len(valid) >= 2:
                    all_times.append(valid)
                    deltas = np.diff(valid)
                    deltas = deltas[deltas > 0]
                    if len(deltas) > 0:
                        all_deltas.append(deltas)

        if len(all_times) == 0:
            logger.warning("[TimeNorm] No valid times found; skip normalization/quantization.")
            return

        flat_times = np.concatenate(all_times)

        # --- Fit time normalizer (optional) ---
        if self.time_normalization in ["standard", "std"]:
            self.time_normalizer = StandardScaler()
            self.time_normalizer.fit(flat_times.reshape(-1, 1))
            logger.info("[TimeNorm] Fitted StandardScaler for time.")
        elif self.time_normalization == "minmax":
            self.time_normalizer = MinMaxScaler(feature_range=(0, 1))
            self.time_normalizer.fit(flat_times.reshape(-1, 1))
            logger.info("[TimeNorm] Fitted MinMaxScaler for time.")
        else:
            self.time_normalizer = None
            logger.info("[TimeNorm] Time normalization disabled (using raw timestamps).")

        # --- Infer quantization resolution ---
        if self.auto_quantize:
            if len(all_deltas) > 0:
                median_delta = np.median(np.concatenate(all_deltas))
                self.quantize_resolution = max(median_delta, 1e-9)
                logger.info(f"[Quantize] Inferred resolution: {self.quantize_resolution:.6f}")
            else:
                self.quantize_resolution = 1.0
                logger.warning("[Quantize] No deltas found, defaulting to resolution=1.0.")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_idx):
        """Load a single time-series sample."""
        item = self.data[seq_idx]

        if isinstance(item, dict):
            sequence = np.array(item["sequence"], dtype=np.float32)
            time = np.array(item["time"], dtype=np.float64)
            mask = np.array(item.get("mask", np.ones_like(sequence)), dtype=int)
        elif isinstance(item, (list, np.ndarray)):
            sequence = np.array(item, dtype=np.float32)
            time = np.arange(len(sequence), dtype=np.float64)
            mask = np.ones(len(sequence), dtype=int)
        else:
            raise TypeError(f"Unsupported item type at index {seq_idx}: {type(item)}")

        # --- Validate lengths ---
        if not (len(sequence) == len(time) == len(mask)):
            raise ValueError(f"[Dataset] Inconsistent lengths at index {seq_idx}")

        # --- Quantization ---
        if self.quantize_resolution is not None:
            time = quantize_time(time, initial_resolution=self.quantize_resolution)

        # --- Optional time normalization ---
        if self.time_normalizer is not None:
            time = self.time_normalizer.transform(time.reshape(-1, 1)).flatten()

        # --- Sequence normalization ---
        if self.data_normalizer is not None:
            sequence = self.data_normalizer.transform(sequence.reshape(-1, 1)).reshape(-1)

        # --- Ensure monotonic time ---
        if not np.all(np.diff(time[mask == 1]) >= 0):
            sort_idx = np.argsort(time)
            sequence, time, mask = sequence[sort_idx], time[sort_idx], mask[sort_idx]
            logger.warning(f"[TimeCheck] Non-monotonic time detected at idx {seq_idx}, sorted.")

        return {
            "sequence": sequence.astype(np.float32),
            "time": time.astype(np.float32),
            "mask": mask.astype(np.int32),
        }

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum(self.get_sequence_length(i) for i in range(len(self)))
        return self.num_tokens

    def get_sequence_length(self, seq_idx):
        item = self.data[seq_idx]
        if isinstance(item, dict) and "sequence" in item:
            return len(item["sequence"])
        elif isinstance(item, (list, np.ndarray)):
            return len(item)
        return 0

    def get_sequence_length_by_idx(self, seq_idx):
        """Required by TimeSeriesDataset abstract base class."""
        return self.get_sequence_length(seq_idx)

    def get_time_normalizer(self):
        return self.time_normalizer
    
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