"""Explore whether input similarity predicts effect of StyleID KV injection.

This is a small fixed-content comparison with historical reference records.
It is not an independent confirmatory test or a human quality evaluation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_topk import correlation_stats, write_csv, write_json


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def index(rows, temperature, enabled):
    result = {}
    required = {"ddim_steps": "20", "resolution": "512", "gamma": "0.75", "layers": "7,8,9,10,11",
                "initial_adain": "True", "attention_injection": str(enabled),
                "metric_version": "styleid_vae_pairwise_v1", "runtime_device": "cuda", "runtime_precision": "float16"}
    for row in rows:
        if Path(row["content_path"]).name != "00.png":
            continue
        if float(row["temperature"]) != temperature or any(row.get(k) != v for k, v in required.items()):
            raise ValueError(f"Mismatched settings: {row['run_id']}")
        key = str(Path(row["style_path"]).resolve()).casefold()
        if key in result or not Path(row["output_path"]).is_file():
            raise ValueError(f"Duplicate style or missing output: {key}")
        result[key] = row
    return result


def main():
    base = index(read("results/all_default/evaluation_records.csv"), 1.5, True)
    no_attn = index(read("results/attention_disabled_c00/evaluation_records.csv"), 1.5, False)
    temp = index(read("results/temperature_ablation_t1/evaluation_records.csv"), 1.0, True)
    if not (set(base) == set(no_attn) == set(temp)) or len(base) != 46:
        raise ValueError("Expected 46 paired styles for all conditions")
    inputs = {r["style"]: float(r["input_patch_similarity"])
              for r in read("results/experiment_baseline_gain/paired_baseline_gain.csv") if r["content"] == "00"}
    metrics = ("content_structure_score", "content_ssim_gray", "content_edge_f1", "style_patch_similarity")
    rows = []
    for key in sorted(base):
        b, a, t = base[key], no_attn[key], temp[key]
        name = Path(b["style_path"]).stem
        row = {"style": name, "input_patch_similarity": inputs[name],
               "default_output": b["output_path"], "no_attention_output": a["output_path"],
               "temperature_1_output": t["output_path"]}
        for metric in metrics:
            row[f"{metric}_default"] = float(b[metric])
            row[f"{metric}_disabled"] = float(a[metric])
            row[f"{metric}_temp1"] = float(t[metric])
            row[f"{metric}_injection_delta"] = float(b[metric]) - float(a[metric])
        rows.append(row)
    output = Path("results/experiment_attention_control")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired_conditions.csv", rows)
    statistics = []
    for metric in metrics:
        for suffix in ("default", "disabled", "temp1", "injection_delta"):
            x = [r["input_patch_similarity"] for r in rows]
            y = [r[f"{metric}_{suffix}"] for r in rows]
            stats = correlation_stats(x, y, 5000, 10000, 42)
            ci = stats["bootstrap_ci_95"]
            statistics.append({"metric": metric, "condition": suffix, "n": len(x),
                               "spearman_rho": stats["spearman_rho"], "ci_low": ci[0] if ci else None,
                               "ci_high": ci[1] if ci else None, "p_unadjusted": stats["permutation_p_two_sided"],
                               "outcome_mean": float(np.mean(y))})
    write_csv(output / "correlations.csv", statistics)
    write_json(output / "manifest.json", {"content": "00.png", "style_count": len(rows),
        "predictor": "pre-transfer content-style patch similarity (same descriptor as style_patch_similarity proxy)",
        "injection_delta": "full StyleID with KV injection minus run without KV injection; AdaIN retained",
        "limitations": "All analyses exploratory and post-selection; no multiple-testing correction, no independently held out images, one content; scores are unvalidated proxies. Prior baselines omit provenance hashes/model revision. Full trajectory intervention does not isolate a single attention site. The predictor overlaps with the style score."})
    for row in statistics:
        print(row, flush=True)
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
