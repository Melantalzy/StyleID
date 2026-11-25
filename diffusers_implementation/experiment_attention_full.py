"""Test a fixed input-similarity/SSIM-injection hypothesis on contents 06-19."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_topk import correlation_stats, write_csv, write_json


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_records(path, enabled):
    expected = {"ddim_steps": "20", "resolution": "512", "gamma": "0.75", "temperature": "1.5",
                "layers": "7,8,9,10,11", "initial_adain": "True", "attention_injection": str(enabled),
                "metric_version": "styleid_vae_pairwise_v1", "runtime_device": "cuda", "runtime_precision": "float16"}
    result = {}
    for r in read(path):
        if any(r.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Mismatched configuration: {r['run_id']}")
        key = (str(Path(r["content_path"]).resolve()).casefold(), str(Path(r["style_path"]).resolve()).casefold())
        if key in result or not Path(r["output_path"]).is_file():
            raise ValueError(f"Duplicate or missing file: {key}")
        result[key] = r
    return result


def main():
    on = load_records("results/all_default/evaluation_records.csv", True)
    off = load_records("results/attention_disabled_c00/evaluation_records.csv", False)
    if set(on) != set(off) or len(on) != 920:
        raise ValueError(f"Expected complete 20x46 design; found {len(on)} and {len(off)}")
    predictors = {(r["content"], r["style"]): float(r["input_patch_similarity"])
                  for r in read("results/experiment_baseline_gain/paired_baseline_gain.csv")}
    pairs = []
    for key in sorted(on):
        a, b = on[key], off[key]
        cid, sid = Path(a["content_path"]).stem, Path(a["style_path"]).stem
        on_score, off_score = float(a["content_ssim_gray"]), float(b["content_ssim_gray"])
        pairs.append({"content": cid, "style": sid, "input_patch_similarity": predictors[(cid, sid)],
                      "ssim_on": on_score, "ssim_off": off_score, "ssim_injection_delta": on_score - off_score,
                      "on_output": a["output_path"], "off_output": b["output_path"]})
    results = []
    for cid in sorted({p["content"] for p in pairs}):
        group = [p for p in pairs if p["content"] == cid]
        stats = correlation_stats([p["input_patch_similarity"] for p in group],
                                  [p["ssim_injection_delta"] for p in group], 5000, 10000, 123)
        ci = stats["bootstrap_ci_95"]
        results.append({"content": cid, "split": "discovery" if int(cid) < 6 else "followup",
                        "n": stats["n"], "rho": stats["spearman_rho"],
                        "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
                        "p_two_sided": stats["permutation_p_two_sided"], "p_holm_followup_14": None,
                        "mean_delta": float(np.mean([p["ssim_injection_delta"] for p in group]))})
    followup = sorted([r for r in results if r["split"] == "followup" and r["p_two_sided"] is not None], key=lambda r: r["p_two_sided"])
    maximum = 0.0
    for i, row in enumerate(followup):
        maximum = min(1.0, max(maximum, row["p_two_sided"] * (len(followup) - i)))
        row["p_holm_followup_14"] = maximum
    out = Path("results/experiment_attention_full")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_conditions.csv", pairs)
    write_csv(out / "hypothesis_results.csv", results)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, row in enumerate(results):
        color = "#3274a1" if row["split"] == "followup" else "#999999"
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, linewidth=1)
        ax.scatter(row["rho"], i, color=color, s=22)
    ax.axvline(0, color="black", linestyle="--", linewidth=.8)
    ax.set_yticks(range(len(results)), [r["content"] for r in results])
    ax.invert_yaxis()
    ax.set(xlabel="Spearman rho (95% bootstrap interval)", ylabel="Content image",
           title="Input patch similarity vs SSIM injection effect\nGray: discovery, blue: follow-up")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "correlation_by_content.png", dpi=150)
    plt.close(fig)
    write_json(out / "manifest.json", {"hypothesis": "positive Spearman association of input patch similarity and SSIM(on)-SSIM(off)",
        "discovery": "contents 00-05", "followup": "contents 06-19; pre-fixed after discovery control results",
        "tests": "two-sided permutation p per content, Holm over 14 followup tests; 5000 paired style bootstraps, 10000 permutations, seed 123",
        "limits": "Same 46 styles in discovery and followup; all input images and default outputs previously analysed; no untouched independent data. Pair output and SSIM not human quality. Historical model/input provenance incomplete. Post-selection overall remains exploratory; no causal attribution to predictor. Holm does not make 14 contents independent replications."})
    for row in results:
        print(row, flush=True)
    positive = sum(r["rho"] > 0 for r in followup)
    corrected = sum(r["p_holm_followup_14"] < 0.05 for r in followup)
    print(f"Follow-up positive {positive}/{len(followup)}, Holm p<.05 {corrected}/{len(followup)}")
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
