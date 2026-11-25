"""Frozen follow-up on new, roughly deduplicated ArtFlow style images."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_topk import correlation_stats, spearman, write_csv, write_json


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def records(path, enabled):
    result = {}
    required = {"ddim_steps": "20", "resolution": "512", "gamma": "0.75", "temperature": "1.5",
                "layers": "7,8,9,10,11", "initial_adain": "True", "attention_injection": str(enabled),
                "metric_version": "styleid_vae_pairwise_v1", "runtime_device": "cuda", "runtime_precision": "float16"}
    for r in read(path):
        if any(r.get(k) != v for k, v in required.items()):
            raise ValueError(f"Config mismatch in {r['run_id']}")
        key = (str(Path(r["content_path"]).resolve()).casefold(), str(Path(r["style_path"]).resolve()).casefold())
        if key in result or not Path(r["output_path"]).is_file():
            raise ValueError(f"Duplicate or missing output: {key}")
        result[key] = r
    return result


def main():
    on = records("results/artflow_inputs_styleid_on/evaluation_records.csv", True)
    off = records("results/artflow_inputs_styleid_off/evaluation_records.csv", False)
    if set(on) != set(off) or len(on) != 500:
        raise ValueError(f"Expected 10x50 on/off records: {len(on)}/{len(off)}")
    full_styles = sorted(Path("downloads/style-transfer-dataset/styles").glob("*"), key=lambda p: p.name)
    initial = {p.stem for p in full_styles[:15]}
    duplicates = {Path(r["image"]).stem for r in read("results/artflow_duplicate_audit.csv")
                  if r["category"] == "sty" and r["possible_duplicate"] == "True"}
    followup = {p.stem for p in full_styles[15:]} - duplicates
    if len(followup) != 28:
        raise ValueError(f"Expected 28 follow-up styles; got {len(followup)}")
    inputs = {(r["content"], r["style"]): float(r["input_patch_similarity"])
              for r in read("results/experiment_artflow_human/pairs.csv")}
    pairs = []
    for key in sorted(on):
        a, b = on[key], off[key]
        cid = Path(a["content_path"]).stem.removeprefix("content_")
        style_stem = Path(a["style_path"]).stem
        if style_stem not in followup:
            continue
        sid = style_stem.removeprefix("style_")
        ss_on, ss_off = float(a["content_ssim_gray"]), float(b["content_ssim_gray"])
        pairs.append({"content": cid, "style": sid, "input_patch_similarity": inputs[(cid, sid)],
                      "ssim_on": ss_on, "ssim_off": ss_off, "ssim_delta": ss_on - ss_off,
                      "on_output": a["output_path"], "off_output": b["output_path"]})
    contents = sorted({r["content"] for r in pairs}, key=int)
    styles = sorted({r["style"] for r in pairs}, key=int)
    if len(contents) != 10 or len(styles) != 28 or len(pairs) != 280:
        raise ValueError("Incomplete 10x28 matrix")
    lookup = {(r["content"], r["style"]): r for r in pairs}
    x = np.array([[lookup[(c, s)]["input_patch_similarity"] for s in styles] for c in contents])
    y = np.array([[lookup[(c, s)]["ssim_delta"] for s in styles] for c in contents])
    groups = []
    for i, content in enumerate(contents):
        stats = correlation_stats(x[i], y[i], 5000, 10000, 4321)
        ci = stats["bootstrap_ci_95"]
        groups.append({"content": content, "n": 28, "rho": stats["spearman_rho"],
                       "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
                       "p_two_sided": stats["permutation_p_two_sided"], "p_holm_10": None})
    ordered = sorted((r for r in groups if r["p_two_sided"] is not None), key=lambda r: r["p_two_sided"])
    value = 0.0
    for i, row in enumerate(ordered):
        value = min(1.0, max(value, row["p_two_sided"] * (len(ordered) - i)))
        row["p_holm_10"] = value
    rng = np.random.default_rng(4321)
    estimates = []
    for _ in range(5000):
        ci = rng.integers(0, len(contents), len(contents))
        si = rng.integers(0, len(styles), len(styles))
        vals = [spearman(x[i, si], y[i, si]) for i in ci]
        vals = [v for v in vals if v is not None]
        if vals:
            estimates.append(float(np.mean(vals)))
    observed = [r["rho"] for r in groups if r["rho"] is not None]
    bounds = [float(v) for v in np.quantile(estimates, [.025, .975])]
    summary = {"n_contents": len(contents), "n_styles": len(styles), "mean_rho": float(np.mean(observed)),
               "positive": sum(r > 0 for r in observed), "negative": sum(r < 0 for r in observed),
               "two_way_bootstrap_ci": bounds, "holm_significant": sum(r["p_holm_10"] < .05 for r in ordered),
               "initial_styles_excluded": sorted(initial), "rough_duplicates_excluded": sorted(duplicates),
               "limitations": "The same ten contents were in the first 14-style test; styles' human ratings were inspected in ArtFlow full dataset. Coarse duplicate audit only. Nonrandom corpus. SSIM is not human rating; hypothesis selected after earlier exploration, no causal claim for input similarity."}
    out = Path("results/experiment_artflow_new_styles")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_conditions.csv", pairs)
    write_csv(out / "per_content.csv", groups)
    write_json(out / "summary.json", summary)
    print(summary, flush=True)
    for row in groups:
        print(row, flush=True)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
