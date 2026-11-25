"""Check one post-discovery hypothesis on content 03-05, same 46 styles.

Hypothesis frozen after inspecting contents 00-02: input patch similarity has
positive association with the change in grayscale SSIM caused by KV injection.
Inputs and baseline images were previously explored; styles are not held out.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_topk import correlation_stats, write_csv, write_json


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def index(path, enabled):
    result = {}
    expected = {"ddim_steps": "20", "resolution": "512", "gamma": "0.75", "temperature": "1.5",
                "layers": "7,8,9,10,11", "initial_adain": "True", "attention_injection": str(enabled),
                "metric_version": "styleid_vae_pairwise_v1", "runtime_device": "cuda", "runtime_precision": "float16"}
    for row in read(path):
        cid, sid = Path(row["content_path"]).stem, Path(row["style_path"]).stem
        if cid not in ("03", "04", "05"):
            continue
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Settings mismatch {row['run_id']}")
        key = (cid, sid)
        if key in result or not Path(row["output_path"]).is_file():
            raise ValueError(f"Duplicate or missing output: {key}")
        result[key] = row
    return result


def main():
    on = index("results/all_default/evaluation_records.csv", True)
    off = index("results/attention_disabled_c00/evaluation_records.csv", False)
    if set(on) != set(off) or len(on) != 138:
        raise ValueError(f"Expected 138 matched pairs: {len(on)} vs {len(off)}")
    predictors = {(r["content"], r["style"]): float(r["input_patch_similarity"])
                  for r in read("results/experiment_baseline_gain/paired_baseline_gain.csv")}
    pairs = []
    for key in sorted(on):
        a, b = on[key], off[key]
        for field in ("content_path", "style_path"):
            if Path(a[field]).resolve() != Path(b[field]).resolve():
                raise ValueError(f"Mismatched {field} for {key}")
        scores = {f"{metric}_{condition}": float(row[metric]) for metric in ("content_ssim_gray", "content_structure_score", "content_edge_f1", "style_patch_similarity") for condition, row in (("on", a), ("off", b))}
        pairs.append({"content": key[0], "style": key[1], "input_patch_similarity": predictors[key],
                      **scores, "ssim_delta": scores["content_ssim_gray_on"] - scores["content_ssim_gray_off"],
                      "on_output": a["output_path"], "off_output": b["output_path"]})
    summaries = []
    for cid in ("03", "04", "05"):
        group = [p for p in pairs if p["content"] == cid]
        stats = correlation_stats([r["input_patch_similarity"] for r in group], [r["ssim_delta"] for r in group], 5000, 10000, 123)
        ci = stats["bootstrap_ci_95"]
        summaries.append({"content": cid, "n": len(group), "rho": stats["spearman_rho"],
                          "ci_low": ci[0], "ci_high": ci[1], "p_two_sided": stats["permutation_p_two_sided"],
                          "p_holm_3": None, "delta_mean": float(np.mean([r["ssim_delta"] for r in group]))})
    sorted_rows = sorted(summaries, key=lambda row: row["p_two_sided"])
    prev = 0.0
    for i, row in enumerate(sorted_rows):
        prev = min(1.0, max(prev, row["p_two_sided"] * (len(sorted_rows) - i)))
        row["p_holm_3"] = prev
    out = Path("results/experiment_attention_replication")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_conditions.csv", pairs)
    write_csv(out / "hypothesis_results.csv", summaries)
    write_json(out / "manifest.json", {"hypothesis": "positive Spearman association between input patch similarity and SSIM(on)-SSIM(off)",
               "selection": "frozen after exploratory contents 00-02; contents 03-05 and the same styles had previous default outputs, and input predictors had been explored",
               "limits": "No new styles, same 46 references; 3 content images; unvalidated SSIM proxy; baseline outputs lack historical hashes/model revision. Holm correction for 3 two-sided group tests; shared styles induce dependence.",
               "bootstrap": 5000, "permutations": 10000, "seed": 123})
    for row in summaries:
        print(row, flush=True)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
