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

    return torch.stack(seqs, dim=0), torch.stack(times, dim=0)

def snap_and_dedup_times(t_scaled, snap=0.1):

    snapped = torch.round(t_scaled / snap) * snap

    # ensure monotonic by shifting small epsilon
    eps = 1e-4
    for i in range(1, snapped.numel()):
        if snapped[0, i] <= snapped[0, i-1]:
            snapped[0, i] = snapped[0, i-1] + eps

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
    future_times = full_scaled_times[:, C:C+P]

    cur_vals = hist_vals.clone()
    cur_times = hist_times.clone()

    preds_norm = []

    for i in range(P):
        inp_vals = cur_vals.unsqueeze(-1)  # [1, L, 1]
        inp_times = cur_times             # [1, L]

        with torch.no_grad():
            out = model(
                input_ids=inp_vals,
                time_values=inp_times,
                next_target_time_values=None,
                return_dict=True,
            )

        next_norm = out.logits[:, -1, :]  # [1, 1]
        preds_norm.append(next_norm.squeeze(0))

        next_t = future_times[:, i:i+1]

        cur_vals = torch.cat([cur_vals, next_norm], dim=1)
        cur_times = torch.cat([cur_times, next_t], dim=1)

    preds_norm = torch.stack(preds_norm, dim=1)
    preds = preds_norm * std + mean  # de-normalize

    return preds.squeeze(0)

def evaluate_nonoverlap(model, seq, times, C, P, mean, std):

    rmse_list, mae_list = [], []

    T = seq.numel()
    w = C + P

    for start in range(0, T, w):
        end = start + w
        if end > T:
            break

        s = seq[start:end]
        t = times[start:end]

        pred = mira_predict_autoreg_norm(model, s.unsqueeze(0), t.unsqueeze(0), C, P, mean, std)
        gt = s[C:C+P]

        rmse_list.append(torch.sqrt(F.mse_loss(pred, gt)).item())
        mae_list.append(F.l1_loss(pred, gt).item())

    return rmse_list, mae_list

def rolling_eval_dataset(model, values, times, settings):

    results = {}

    for (C, P) in settings:
        print(f"\n===== Evaluating: history={C}, pred={P} =====")
        all_rmse, all_mae = [], []

        for i in range(values.size(0)):
            seq = values[i]
            tms = times[i]

            if seq.size(0) < C + P:
                continue

            mean = seq.mean()
            std = seq.std() + 1e-6

            rmses, maes = evaluate_nonoverlap(model, seq, tms, C, P, mean, std)
            all_rmse.extend(rmses)
            all_mae.extend(maes)

        results[(C, P)] = dict(
            rmse=sum(all_rmse) / len(all_rmse),
            mae=sum(all_mae) / len(all_mae),
            n=len(all_rmse),
        )

        print(f"RMSE={results[(C,P)]['rmse']:.4f} | "
              f"MAE={results[(C,P)]['mae']:.4f} | "
              f"N={results[(C,P)]['n']}")

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
    values, times = load_jsonl_timeseries(args.data)
    print("Values:", values.shape, "Times:", times.shape)

    settings = [
        (48, 24),
        (72, 36),
        (96, 48),
        (128, 64),
    ]

    results = rolling_eval_dataset(model, values, times, settings)

    print("\n===== FINAL SUMMARY =====")
    for (C, P), info in results.items():
        print(f"{C}->{P}: RMSE={info['rmse']:.4f}, MAE={info['mae']:.4f}, N={info['n']}")


if __name__ == "__main__":
    main()
