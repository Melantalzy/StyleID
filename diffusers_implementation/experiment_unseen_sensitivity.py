"""Exploratory decomposition of held-out sample similarity associations."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_baseline_gain import paired_patch_similarity
from experiment_local_matching import image_descriptors, match
from experiment_topk import spearman, write_csv, write_json
from run_styleid_diffusers import load_input_image
from single_image_metrics import _color_histogram_distance, _patch_descriptors


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    def indexed(path):
        return {(Path(r["content_path"]).stem, Path(r["style_path"]).stem): r for r in read(path)}
    on = indexed("results/external_eval_on/evaluation_records.csv")
    off = indexed("results/external_eval_off/evaluation_records.csv")
    if set(on) != set(off) or len(on) != 117:
        raise ValueError("Incomplete data")
    paths = {str(Path(r[f]).resolve()): Path(r[f]).resolve() for r in on.values() for f in ("content_path", "style_path")}
    imgs = {p: load_input_image(p, 512) for p in paths}
    desc = {p: _patch_descriptors(image) for p, image in imgs.items()}
    hog = {p: image_descriptors(p, 512) for p in paths}
    rows = []
    for key in sorted(on):
        a, b = on[key], off[key]
        cp, sp = str(Path(a["content_path"]).resolve()), str(Path(a["style_path"]).resolve())
        _, _, structure = match(hog[cp], hog[sp])
        if structure is None:
            raise ValueError(f"Too few active patches: {key}")
        metrics = {name + "_delta": float(a[name]) - float(b[name]) for name in
                   ("content_ssim_gray", "content_edge_f1", "content_structure_score", "style_patch_similarity")}
        rows.append({"content": key[0], "style": key[1],
                     "patch_similarity": paired_patch_similarity(desc[cp], desc[sp]),
                     "color_similarity": 1 - _color_histogram_distance(imgs[cp], imgs[sp]),
                     "structure_match": structure, **metrics})
    contents = sorted({r["content"] for r in rows})
    styles = sorted({r["style"] for r in rows})
    lookup = {(r["content"], r["style"]): r for r in rows}
    results, rng = [], np.random.default_rng(551)
    for feature in ("patch_similarity", "color_similarity", "structure_match"):
        for target in ("content_ssim_gray_delta", "content_edge_f1_delta", "content_structure_score_delta", "style_patch_similarity_delta"):
            x = np.array([[lookup[(c, s)][feature] for s in styles] for c in contents])
            y = np.array([[lookup[(c, s)][target] for s in styles] for c in contents])
            rhos = [spearman(a, b) for a, b in zip(x, y)]
            estimates = []
            for _ in range(1500):
                ci = rng.integers(0, len(contents), len(contents))
                si = rng.integers(0, len(styles), len(styles))
                r = [spearman(x[i, si], y[i, si]) for i in ci]
                r = [v for v in r if v is not None]
                if r:
                    estimates.append(float(np.mean(r)))
            low, high = np.quantile(estimates, [.025, .975])
            results.append({"feature": feature, "target": target, "mean_rho": float(np.mean(rhos)),
                            "ci_low": float(low), "ci_high": float(high),
                            "positive_contents": sum(r > 0 for r in rhos), "contents": len(rhos)})
            print(results[-1], flush=True)
    out = Path("results/experiment_unseen_sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pair_features.csv", rows)
    write_csv(out / "sensitivity_summary.csv", results)
    write_json(out / "manifest.json", {"status": "post-hoc exploratory; no multiplicity adjustment; not a confirmation",
               "bootstrap": "1500 crossed resamples of 9 contents and 13 shared styles",
               "limitations": "Shared image statistics, very small sample; none of these proxy outcomes is human-rated quality. Similar features can correlate due to metric design rather than mechanism."})
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
