# Copyright (c) Microsoft
# Licensed under MIT

import torch
import torch.nn.functional as F
import json
import argparse

from MIRA.mira.models.modeling_mira import MIRAForPrediction
from MIRA.mira.models.utils_time_normalization import normalize_time_for_ctrope


def load_jsonl_timeseries(jsonl_path):
    seqs, times = [], []
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            seqs.append(torch.tensor(obj["sequence"], dtype=torch.float32))
            times.append(torch.tensor(obj["time"], dtype=torch.float32))
    return seqs, times  # keep as lists (no stack!)


def snap_and_dedup_times(t_scaled, snap=0.1):
    snapped = torch.round(t_scaled / snap) * snap
    eps = 1e-4
    for i in range(1, snapped.numel()):
        if snapped[0, i] <= snapped[0, i - 1]:
            snapped[0, i] = snapped[0, i - 1] + eps
    return snapped


def mira_predict_autoreg_norm(model, values, raw_times, C, P, mean, std):
    device = next(model.parameters()).device

    values = values.to(device)
    raw_times = raw_times.to(device)
    mean = mean.to(device)
    std = std.to(device)

    values_norm = (values - mean) / std

    full_scaled_times, t_min, t_max = normalize_time_for_ctrope(
        time_values=raw_times,
        attention_mask=torch.ones_like(raw_times),
        seq_length=raw_times.shape[1],
        alpha=1.0,
    )

    full_scaled_times = snap_and_dedup_times(full_scaled_times)

    hist_vals = values_norm[:, :C]
    hist_times = full_scaled_times[:, :C]
    future_times = full_scaled_times[:, C:C + P]

    cur_vals = hist_vals.clone()
    cur_times = hist_times.clone()

    preds_norm = []

    for i in range(P):
        inp_vals = cur_vals.unsqueeze(-1)
        inp_times = cur_times

        with torch.no_grad():
            out = model(
                input_ids=inp_vals,
                time_values=inp_times,
                next_target_time_values=None,
                return_dict=True,
            )

        next_norm = out.logits[:, -1, :]
        preds_norm.append(next_norm.squeeze(0))

        next_t = future_times[:, i:i + 1]
        cur_vals = torch.cat([cur_vals, next_norm], dim=1)
        cur_times = torch.cat([cur_times, next_t], dim=1)

    preds_norm = torch.stack(preds_norm, dim=1)
    preds = preds_norm * std + mean
    return preds.squeeze(0)

def evaluate_one_window(model, seq, times, C, P, mean, std):
    """Evaluate only one window (batch size = 1)."""
    device = next(model.parameters()).device

    T = len(seq)
    if T < C + P:
        return None, None

    # Move sequence and time to device
    hist = seq[:C + P].to(device)
    t_hist = times[:C + P].to(device)

    mean = mean.to(device)
    std = std.to(device)

    pred = mira_predict_autoreg_norm(
        model,
        hist.unsqueeze(0),
        t_hist.unsqueeze(0),
        C,
        P,
        mean,
        std,
    )

    gt = hist[C:C + P].to(device)

    rmse = torch.sqrt(F.mse_loss(pred, gt)).item()
    mae = F.l1_loss(pred, gt).item()
    return rmse, mae


def rolling_eval_dataset(model, seq_list, time_list, settings):

    results = {}

    for C, P in settings:
        rmses, maes = [], []

        for seq, tms in zip(seq_list, time_list):

            device = next(model.parameters()).device
            mean = seq.mean().to(device)
            std = (seq.std() + 1e-6).to(device)

            rmse, mae = evaluate_one_window(model, seq, tms, C, P, mean, std)
            if rmse is not None:
                rmses.append(rmse)
                maes.append(mae)

        results[(C, P)] = {
            "rmse": sum(rmses) / len(rmses) if rmses else float("nan"),
            "mae": sum(maes) / len(maes) if maes else float("nan"),
            "n": len(rmses),
        }

        print(f"{C}->{P} | N={len(rmses)} | RMSE={results[(C,P)]['rmse']:.4f} | MAE={results[(C,P)]['mae']:.4f}")

    return results


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    print("[INFO] Loading model:", args.model)
    model = MIRAForPrediction.from_pretrained(args.model).cuda()
    model.eval()

    print("[INFO] Loading dataset:", args.data)
    seq_list, time_list = load_jsonl_timeseries(args.data)
    print("Loaded:", len(seq_list), "series")

    settings = [
        (48, 24),
        (72, 36),
        (96, 48),
        (128, 64),
    ]

    results = rolling_eval_dataset(model, seq_list, time_list, settings)

    print("\n===== FINAL SUMMARY =====")
    for (C, P), info in results.items():
        print(f"{C}->{P}: RMSE={info['rmse']:.4f}, MAE={info['mae']:.4f}, N={info['n']}")


if __name__ == "__main__":
    main()
