"""Explore pairwise predictors of attention-injection effects across 3 contents."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_topk import correlation_stats, write_csv, write_json


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def make_index(path, enabled):
    records = read(path)
    required = {"ddim_steps": "20", "resolution": "512", "gamma": "0.75", "temperature": "1.5",
                "layers": "7,8,9,10,11", "initial_adain": "True", "attention_injection": str(enabled),
                "metric_version": "styleid_vae_pairwise_v1", "runtime_device": "cuda", "runtime_precision": "float16"}
    result = {}
    for row in records:
        content = Path(row["content_path"]).stem
        if content not in ("00", "01", "02"):
            continue
        if any(row.get(k) != v for k, v in required.items()):
            raise ValueError(f"Mismatched settings: {row['run_id']}")
        key = (str(Path(row["content_path"]).resolve()).casefold(), str(Path(row["style_path"]).resolve()).casefold())
        if key in result or not Path(row["output_path"]).is_file():
            raise ValueError(f"Duplicate or missing output: {key}")
        result[key] = row
    return result


def main():
    default = make_index("results/all_default/evaluation_records.csv", True)
    disabled = make_index("results/attention_disabled_c00/evaluation_records.csv", False)
    if set(default) != set(disabled) or len(default) != 138:
        raise ValueError(f"Expected 138 paired results; got {len(default)}/{len(disabled)}")
    inputs = {(r["content"], r["style"]): float(r["input_patch_similarity"])
              for r in read("results/experiment_baseline_gain/paired_baseline_gain.csv")}
    structure = {(r["content"], r["style"]): float(r["match_active"])
                 for r in read("results/experiment_local_matching/pair_metrics.csv") if r["match_active"]}
    metrics = ("content_structure_score", "content_ssim_gray", "content_edge_f1", "style_patch_similarity")
    rows = []
    for key in sorted(default):
        b, a = default[key], disabled[key]
        image_id = (Path(b["content_path"]).stem, Path(b["style_path"]).stem)
        row = {"content": image_id[0], "style": image_id[1],
               "input_patch_similarity": inputs[image_id], "input_structure_match": structure[image_id],
               "default_output": b["output_path"], "no_attention_output": a["output_path"]}
        for metric in metrics:
            row[metric + "_on"] = float(b[metric])
            row[metric + "_off"] = float(a[metric])
            row[metric + "_delta"] = float(b[metric]) - float(a[metric])
        rows.append(row)
    out = Path("results/experiment_attention_control_multi")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_conditions.csv", rows)
    summary = []
    for feature in ("input_patch_similarity", "input_structure_match"):
        for content in ("00", "01", "02"):
            items = [r for r in rows if r["content"] == content]
            for metric in metrics:
                x = [r[feature] for r in items]
                y = [r[metric + "_delta"] for r in items]
                result = correlation_stats(x, y, 5000, 10000, 42)
                ci = result["bootstrap_ci_95"]
                summary.append({"feature": feature, "content": content, "metric": metric, "n": result["n"],
                                "rho": result["spearman_rho"], "ci_low": ci[0] if ci else None,
                                "ci_high": ci[1] if ci else None, "p_unadjusted": result["permutation_p_two_sided"],
                                "q_bh_24": None, "mean_injection_delta": float(np.mean(y))})
    valid = sorted((r for r in summary if r["p_unadjusted"] is not None), key=lambda r: r["p_unadjusted"])
    adj = 1.0
    for i in range(len(valid) - 1, -1, -1):
        adj = min(adj, valid[i]["p_unadjusted"] * len(valid) / (i + 1))
        valid[i]["q_bh_24"] = adj
    write_csv(out / "correlations.csv", summary)
    write_json(out / "manifest.json", {"content_count": 3, "style_count": 46, "pairs": len(rows),
        "injection_delta": "full StyleID minus generation without style KV injection; initial AdaIN retained",
        "metrics": metrics, "features": ["input_patch_similarity", "input_structure_match"],
        "bootstrap": "5000 paired resamples of styles per content", "permutations": 10000,
        "multiplicity": "BH across 24 exploratory group analyses; dependent images and post-selection remain",
        "limitations": "Not an independent replication: all input images and some outputs previously inspected. Only 3 content images and 46 shared styles. Reference quality metrics unvalidated, patch feature overlaps style_patch output score; no causal claim about predictors. Historical baseline inputs and model/code hashes absent."})
    for row in summary:
        print(row, flush=True)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
