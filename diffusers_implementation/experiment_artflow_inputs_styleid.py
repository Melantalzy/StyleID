"""Validate frozen input-similarity hypothesis on ArtFlow dataset inputs.

StyleID on/off outputs are newly generated. ArtFlow ratings apply only to
ArtFlow outputs and are shown solely as an exploratory cross-method comparison.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_local_matching import image_descriptors, match
from experiment_topk import correlation_stats, spearman, write_csv, write_json


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def indexed(path, enabled):
    expected = {"ddim_steps": "20", "resolution": "512", "gamma": "0.75", "temperature": "1.5",
                "layers": "7,8,9,10,11", "initial_adain": "True", "attention_injection": str(enabled),
                "metric_version": "styleid_vae_pairwise_v1", "runtime_device": "cuda", "runtime_precision": "float16"}
    indexed_rows = {}
    for row in read(path):
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Unexpected config for {row['run_id']}")
        key = (str(Path(row["content_path"]).resolve()).casefold(), str(Path(row["style_path"]).resolve()).casefold())
        if key in indexed_rows or not Path(row["output_path"]).is_file():
            raise ValueError(f"Duplicate or missing output {key}")
        indexed_rows[key] = row
    return indexed_rows


def crossed_ci(x, y, seed=90, bootstrap=3000):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(bootstrap):
        ci = rng.integers(0, x.shape[0], x.shape[0])
        si = rng.integers(0, x.shape[1], x.shape[1])
        vals = [spearman(x[i, si], y[i, si]) for i in ci]
        vals = [v for v in vals if v is not None]
        if vals:
            results.append(float(np.mean(vals)))
    return [float(v) for v in np.quantile(results, [.025, .975])]


def main():
    on = indexed("results/artflow_inputs_styleid_on/evaluation_records.csv", True)
    off = indexed("results/artflow_inputs_styleid_off/evaluation_records.csv", False)
    if set(on) != set(off) or len(on) != 150:
        raise ValueError(f"Expected 150 on/off pairs, got {len(on)}/{len(off)}")
    human = {(r["content"], r["style"]): r for r in read("results/experiment_artflow_human/pairs.csv")}
    paths = {str(Path(r[field]).resolve()): Path(r[field]).resolve()
             for r in on.values() for field in ("content_path", "style_path")}
    local_features = {key: image_descriptors(key, 512) for key in paths}
    rows = []
    for key in sorted(on):
        a, b = on[key], off[key]
        cid = Path(a["content_path"]).stem.removeprefix("content_")
        sid = Path(a["style_path"]).stem.removeprefix("style_")
        if sid == "14":
            continue  # Near-duplicate of data_vis/sty/udnie.png.
        record = human[(cid, sid)]
        cp, sp = str(Path(a["content_path"]).resolve()), str(Path(a["style_path"]).resolve())
        _, _, structure = match(local_features[cp], local_features[sp])
        if structure is None:
            raise ValueError(f"Too few active patches: {cid}")
        row = {"content": cid, "style": sid, "input_patch_similarity": float(record["input_patch_similarity"]),
               "input_structure_match": structure,
               "artflow_mean_rating": float(record["mean_rating"]),
               "on_output": a["output_path"], "off_output": b["output_path"]}
        for metric in ("content_ssim_gray", "content_structure_score", "content_edge_f1", "style_patch_similarity"):
            row[metric + "_on"] = float(a[metric])
            row[metric + "_off"] = float(b[metric])
            row[metric + "_delta"] = float(a[metric]) - float(b[metric])
        rows.append(row)
    contents = sorted({r["content"] for r in rows}, key=int)
    styles = sorted({r["style"] for r in rows}, key=int)
    if len(contents) != 10 or len(styles) != 14 or len(rows) != 140:
        raise ValueError("Expected 10x14 matrix after near-duplicate exclusion")
    lookup = {(r["content"], r["style"]): r for r in rows}
    summaries, per_content = [], []
    for feature, target in [("input_patch_similarity", target) for target in ("content_ssim_gray_delta", "content_structure_score_delta", "content_edge_f1_delta", "style_patch_similarity_delta", "artflow_mean_rating")] + [("input_structure_match", target) for target in ("content_ssim_gray_delta", "content_structure_score_delta", "content_edge_f1_delta", "style_patch_similarity_delta", "artflow_mean_rating")]:
        x = np.array([[lookup[(c, s)][feature] for s in styles] for c in contents])
        y = np.array([[lookup[(c, s)][target] for s in styles] for c in contents])
        rhos = [spearman(a, b) for a, b in zip(x, y)]
        ci = crossed_ci(x, y)
        summaries.append({"feature": feature, "target": target, "purpose": "fixed hypothesis" if feature == "input_patch_similarity" and target == "content_ssim_gray_delta" else "exploratory",
                          "n_content": len(contents), "n_style": len(styles), "mean_rho": float(np.mean(rhos)),
                          "positive": sum(r > 0 for r in rhos), "negative": sum(r < 0 for r in rhos),
                          "ci_low": ci[0], "ci_high": ci[1]})
        for i, content in enumerate(contents):
            results = correlation_stats(x[i], y[i], 3000, 5000, 90)
            interval = results["bootstrap_ci_95"]
            per_content.append({"feature": feature, "target": target, "content": content, "n": results["n"], "rho": results["spearman_rho"],
                                "ci_low": interval[0] if interval else None, "ci_high": interval[1] if interval else None,
                                "p_unadjusted": results["permutation_p_two_sided"]})
    out = Path("results/experiment_artflow_inputs_styleid")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_conditions.csv", rows)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "per_content.csv", per_content)
    write_json(out / "manifest.json", {"hypothesis": "positive input patch similarity association with StyleID SSIM(on)-SSIM(off)",
        "sampling": "first 10 contents and 15 styles in sorted filename order; style_14 excluded as near duplicate of prior data_vis udnie, discovered by 64x64 coarse MAE check before testing",
        "n_content": len(contents), "n_style": len(styles), "n_pairs": len(rows),
        "limitations": "Same source images as previously analysed ArtFlow human ratings; not independent of those ratings. ArtFlow rating is not a StyleID rating. Content/style were center-cropped 512 for StyleID but ArtFlow originals use different sizes. Small, nonrandom subset. Additional metrics are exploratory; no multiplicity correction. No human ratings on StyleID outputs.",
        "bootstrap": "3000 crossed resamples of contents and styles; seed 90; content-wise Spearman correlations averaged"})
    for row in summaries:
        print(row, flush=True)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
