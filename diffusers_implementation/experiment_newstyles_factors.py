"""Post-hoc input-factor sensitivity on the 10x28 new-style follow-up."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_artflow_new_styles import read
from experiment_local_matching import image_descriptors, match
from experiment_topk import spearman, write_csv, write_json
from run_styleid_diffusers import load_input_image
from single_image_metrics import _color_histogram_distance


def main():
    pairs = read("results/experiment_artflow_new_styles/paired_conditions.csv")
    if len(pairs) != 280:
        raise ValueError("Expected 10x28 follow-up matrix")
    def index(path):
        return {(Path(r["content_path"]).stem.removeprefix("content_"),
                 Path(r["style_path"]).stem.removeprefix("style_")): r for r in read(path)}
    on = index("results/artflow_inputs_styleid_on/evaluation_records.csv")
    off = index("results/artflow_inputs_styleid_off/evaluation_records.csv")
    root = Path("downloads/style-transfer-dataset")
    contents = sorted({r["content"] for r in pairs}, key=int)
    styles = sorted({r["style"] for r in pairs}, key=int)
    cpaths = {c: root / "contents" / f"content_{c}.jpg" for c in contents}
    spaths = {s: root / "styles" / f"style_{s}.jpg" for s in styles}
    paths = [*cpaths.values(), *spaths.values()]
    images = {str(p): load_input_image(str(p), 512) for p in paths}
    features = {str(p): image_descriptors(p, 512) for p in paths}
    rows = []
    for r in pairs:
        c, s = r["content"], r["style"]
        cp, sp = str(cpaths[c]), str(spaths[s])
        _, _, structure = match(features[cp], features[sp])
        if structure is None:
            raise ValueError(f"Not enough active content patches: {c}")
        differences = {metric + "_delta": float(on[(c, s)][metric]) - float(off[(c, s)][metric])
                       for metric in ("content_ssim_gray", "content_edge_f1", "content_structure_score", "style_patch_similarity")}
        rows.append({"content": c, "style": s, "patch_similarity": float(r["input_patch_similarity"]),
                     "structure_match": structure,
                     "color_similarity": 1.0 - _color_histogram_distance(images[cp], images[sp]), **differences})
    lookup = {(r["content"], r["style"]): r for r in rows}
    rng = np.random.default_rng(751)
    summaries = []
    for feature in ("patch_similarity", "structure_match", "color_similarity"):
        for target in ("content_ssim_gray_delta", "content_edge_f1_delta", "content_structure_score_delta", "style_patch_similarity_delta"):
            x = np.array([[lookup[(c, s)][feature] for s in styles] for c in contents])
            y = np.array([[lookup[(c, s)][target] for s in styles] for c in contents])
            rho = [spearman(a, b) for a, b in zip(x, y)]
            estimates = []
            for _ in range(2000):
                ci = rng.integers(0, len(contents), len(contents))
                si = rng.integers(0, len(styles), len(styles))
                values = [spearman(x[i, si], y[i, si]) for i in ci]
                values = [v for v in values if v is not None]
                if values:
                    estimates.append(float(np.mean(values)))
            low, high = np.quantile(estimates, [.025, .975])
            summaries.append({"feature": feature, "target": target, "mean_rho": float(np.mean(rho)),
                              "positive": sum(v > 0 for v in rho), "negative": sum(v < 0 for v in rho),
                              "ci_low": float(low), "ci_high": float(high), "n_contents": len(contents),
                              "n_styles": len(styles)})
            print(summaries[-1], flush=True)
    out = Path("results/experiment_newstyles_factors")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pair_features.csv", rows)
    write_csv(out / "summary.csv", summaries)
    write_json(out / "manifest.json", {"status": "post-hoc exploratory, 12 comparisons, no multiplicity correction",
        "limits": "10 contents and 28 styles, nonrandom shared samples; metrics are not human ratings. Scores share low-level image statistics; not evidence of causality. The frozen pre-specified test is reported separately in experiment_artflow_new_styles."})
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
