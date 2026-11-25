"""Test frozen similarity-vs-SSIM injection hypothesis on data_vis pairs."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_baseline_gain import paired_patch_similarity
from experiment_style_gradient import sha256
from experiment_topk import correlation_stats, spearman, write_csv, write_json
from run_styleid_diffusers import load_input_image
from single_image_metrics import _patch_descriptors


def load_records(path, enabled):
    required = {"ddim_steps": "20", "resolution": "512", "gamma": "0.75", "temperature": "1.5",
                "layers": "7,8,9,10,11", "initial_adain": "True", "attention_injection": str(enabled),
                "metric_version": "styleid_vae_pairwise_v1", "runtime_device": "cuda", "runtime_precision": "float16"}
    result = {}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if any(row.get(k) != v for k, v in required.items()):
                raise ValueError(f"Mismatched config in {row['run_id']}")
            key = (str(Path(row["content_path"]).resolve()).casefold(), str(Path(row["style_path"]).resolve()).casefold())
            if key in result or not Path(row["output_path"]).is_file():
                raise ValueError(f"Duplicate or missing output: {key}")
            result[key] = row
    return result


def main():
    on = load_records("results/external_eval_on/evaluation_records.csv", True)
    off = load_records("results/external_eval_off/evaluation_records.csv", False)
    if set(on) != set(off) or len(on) != 117:
        raise ValueError(f"Expected 9 x 13 paired results, got {len(on)} and {len(off)}")
    paths = {str(Path(row[field]).resolve()).casefold(): Path(row[field]).resolve()
             for row in on.values() for field in ("content_path", "style_path")}
    descriptors = {k: _patch_descriptors(load_input_image(str(v), 512)) for k, v in paths.items()}
    pairs = []
    for key in sorted(on):
        a, b = on[key], off[key]
        content, style = Path(a["content_path"]).stem, Path(a["style_path"]).stem
        input_match = paired_patch_similarity(descriptors[key[0]], descriptors[key[1]])
        x, y = float(a["content_ssim_gray"]), float(b["content_ssim_gray"])
        pairs.append({"content": content, "style": style, "input_patch_similarity": input_match,
                      "ssim_on": x, "ssim_off": y, "ssim_injection_delta": x - y,
                      "on_output": a["output_path"], "off_output": b["output_path"]})
    contents = sorted({p["content"] for p in pairs})
    styles = sorted({p["style"] for p in pairs})
    if len(contents) != 9 or len(styles) != 13 or len({(p['content'], p['style']) for p in pairs}) != 117:
        raise ValueError("Not a complete 9 x 13 image matrix")
    lookup = {(p["content"], p["style"]): p for p in pairs}
    x = np.array([[lookup[(c, s)]["input_patch_similarity"] for s in styles] for c in contents])
    y = np.array([[lookup[(c, s)]["ssim_injection_delta"] for s in styles] for c in contents])
    summary = []
    for i, content in enumerate(contents):
        r = correlation_stats(x[i], y[i], 5000, 10000, 321)
        ci = r["bootstrap_ci_95"]
        summary.append({"content": content, "n": 13, "rho": r["spearman_rho"],
                        "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
                        "p_two_sided": r["permutation_p_two_sided"], "p_holm_9": None})
    valid = sorted((r for r in summary if r["p_two_sided"] is not None), key=lambda r: r["p_two_sided"])
    maximum = 0.0
    for i, r in enumerate(valid):
        maximum = min(1.0, max(maximum, r["p_two_sided"] * (len(valid) - i)))
        r["p_holm_9"] = maximum
    rhos = [r["rho"] for r in summary if r["rho"] is not None]
    mean_rho = float(np.mean(rhos))
    rng = np.random.default_rng(321)
    estimates = []
    for _ in range(3000):
        ci = rng.integers(0, len(contents), len(contents))
        si = rng.integers(0, len(styles), len(styles))
        replicate = [spearman(x[i, si], y[i, si]) for i in ci]
        replicate = [r for r in replicate if r is not None]
        if replicate:
            estimates.append(float(np.mean(replicate)))
    bounds = [float(v) for v in np.quantile(estimates, [.025, .975])]
    out = Path("results/experiment_unseen_validation")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_conditions.csv", pairs)
    write_csv(out / "per_content.csv", summary)
    write_json(out / "summary.json", {"hypothesis": "positive association between input patch similarity and SSIM(on)-SSIM(off)",
        "mean_rho": mean_rho, "two_way_bootstrap_95_ci": bounds, "positive": sum(r > 0 for r in rhos),
        "negative": sum(r < 0 for r in rhos), "n_contents": len(contents), "n_styles": len(styles),
        "holm_below_0_05": sum(r["p_holm_9"] is not None and r["p_holm_9"] < .05 for r in summary),
        "bootstrap": "3000 resamples of whole content and style indices; styles shared across contents",
        "input_sha256": {str(p): sha256(p) for p in paths.values()},
        "limitations": "9 contents and 13 styles are repository demonstration images, not independently sampled benchmark; coarse 64x64 image check is not a complete duplication audit. Grayscale SSIM and input descriptors share low-level structure/contrast signals; association is not human quality or causal effect of predictor. No independent human judgments."})
    print(f"mean rho {mean_rho:.3f}, CI {bounds}, positive {sum(r > 0 for r in rhos)}/{len(contents)}")
    for r in summary:
        print(r)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
