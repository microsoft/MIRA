import torch
import numpy as np

def time_aware_collate_fn(batch, pad_value=0.0):
    """
    Collate function for TimeAwareDataset samples.
    Pads variable-length sequences in the batch and preserves per-sample time alignment.

    Each item in batch should contain:
        {
            'input_ids': 1D np.array,
            'time_values': 1D np.array,
            'attention_mask': 1D np.array,
            'labels': 1D np.array,
            'loss_mask': 1D np.array,
            'next_target_time_value': scalar (float)
        }
    """
    # Filter out invalid / None samples
    batch = [b for b in batch if b is not None and b.get("input_ids") is not None]
    if len(batch) == 0:
        return {}

    # Determine max sequence length in the batch
    max_len = max(len(b["input_ids"]) for b in batch)

    # Prepare padded containers
    padded_batch = {
        "input_ids": [],
        "time_values": [],
        "attention_mask": [],
        "labels": [],
        "loss_mask": [],
        "next_target_time_value": []
    }

    for item in batch:
        L = len(item["input_ids"])
        pad_len = max_len - L

        # Determine pad time value: use last valid timestamp instead of 0.0
        pad_time_val = (
            float(item["time_values"][-1])
            if len(item["time_values"]) > 0
            else 0.0
        )

        # Pad all arrays
        padded_batch["input_ids"].append(
            np.pad(item["input_ids"], (0, pad_len), constant_values=pad_value)
        )
        padded_batch["time_values"].append(
            np.pad(item["time_values"], (0, pad_len), constant_values=pad_time_val)
        )
        padded_batch["attention_mask"].append(
            np.pad(item["attention_mask"], (0, pad_len), constant_values=0)
        )
        padded_batch["labels"].append(
            np.pad(item["labels"], (0, pad_len), constant_values=pad_value)
        )
        padded_batch["loss_mask"].append(
            np.pad(item["loss_mask"], (0, pad_len), constant_values=0)
        )

        # The next target time should correspond to the *next* timestamp after the last valid time
        if "next_target_time_value" in item:
            padded_batch["next_target_time_value"].append(item["next_target_time_value"])
        else:
            # fallback: use extrapolated next time (delta of last step)
            if len(item["time_values"]) >= 2:
                delta = item["time_values"][-1] - item["time_values"][-2]
                padded_batch["next_target_time_value"].append(item["time_values"][-1] + delta)
            else:
                padded_batch["next_target_time_value"].append(item["time_values"][-1] + 1.0)

    # Convert to PyTorch tensors
    collated_batch = {
        "input_ids": torch.tensor(np.stack(padded_batch["input_ids"]), dtype=torch.float32),
        "time_values": torch.tensor(np.stack(padded_batch["time_values"]), dtype=torch.float32),
        "attention_mask": torch.tensor(np.stack(padded_batch["attention_mask"]), dtype=torch.long),
        "labels": torch.tensor(np.stack(padded_batch["labels"]), dtype=torch.float32),
        "loss_mask": torch.tensor(np.stack(padded_batch["loss_mask"]), dtype=torch.bool),
        "next_target_time_value": torch.tensor(padded_batch["next_target_time_value"], dtype=torch.float32),
    }

    return collated_batch