# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np

def time_aware_collate_fn(batch, pad_value=0, pad_time_value=0.0):
    """
    Collates data from TimeAwareWindowDataset, padding sequences to max length in batch.
    """
    # Find max length in batch for input_ids (which determines length for others)
    max_len = 0
    for item in batch:
        # Handle potential None items if __getitem__ decided to return None
        if item is None or item['input_ids'] is None: continue
        max_len = max(max_len, len(item['input_ids']))

    # Filter out None items
    batch = [item for item in batch if item is not None and item['input_ids'] is not None]
    if not batch: # Return empty dict if batch becomes empty
        return {}

    # Pad each item in the batch
    padded_batch = {
        'input_ids': [],
        'time_values': [],
        'attention_mask': [],
        'labels': [],
        'loss_mask': [],
        'next_target_time_value': []
    }

    for item in batch:
        current_len = len(item['input_ids'])
        padding_length = max_len - current_len

        if padding_length < 0:
             # This shouldn't happen if max_len is calculated correctly
             raise ValueError(f"Negative padding length calculated: {padding_length}")

        # Pad arrays
        padded_batch['input_ids'].append(np.pad(item['input_ids'], (0, padding_length), 'constant', constant_values=pad_value))
        padded_batch['time_values'].append(np.pad(item['time_values'], (0, padding_length), 'constant', constant_values=pad_time_value))
        padded_batch['attention_mask'].append(np.pad(item['attention_mask'], (0, padding_length), 'constant', constant_values=0)) # Pad attention mask with 0
        padded_batch['labels'].append(np.pad(item['labels'], (0, padding_length), 'constant', constant_values=pad_value))
        padded_batch['loss_mask'].append(np.pad(item['loss_mask'], (0, padding_length), 'constant', constant_values=0)) # Pad loss mask with 0
        padded_batch['next_target_time_value'].append(item['next_target_time_value']) # This is scalar, just collect

    # Stack arrays into tensors
    collated_batch = {}
    try:
        collated_batch['input_ids'] = torch.from_numpy(np.stack(padded_batch['input_ids'])).float() # Model expects float input_ids
        collated_batch['time_values'] = torch.from_numpy(np.stack(padded_batch['time_values'])).float()
        collated_batch['attention_mask'] = torch.from_numpy(np.stack(padded_batch['attention_mask'])).long() # Mask is usually Long or Bool
        collated_batch['labels'] = torch.from_numpy(np.stack(padded_batch['labels'])).float()
        collated_batch['loss_mask'] = torch.from_numpy(np.stack(padded_batch['loss_mask'])).bool() # Use bool for loss mask
        collated_batch['next_target_time_value'] = torch.tensor(padded_batch['next_target_time_value'], dtype=torch.float32)
    except Exception as e:
         print("Error during tensor conversion in collate_fn:")
         for key, val_list in padded_batch.items():
             if isinstance(val_list, list) and val_list:
                  print(f"  Shape of first item in '{key}': {np.array(val_list[0]).shape}")
             else:
                  print(f"  Value for '{key}': {val_list}")
         raise e


    return collated_batch

