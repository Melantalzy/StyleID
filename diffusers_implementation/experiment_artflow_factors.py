"""Exploratory input-factor comparisons on human-rated ArtFlow outputs."""
from __future__ import annotations

from pathlib import Path
import csv

import cv2
import numpy as np

from experiment_baseline_gain import cluster_bootstrap_interaction
from experiment_local_matching import image_descriptors, match
from experiment_topk import spearman, write_csv, write_json
from run_styleid_diffusers import load_input_image


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def hist(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    h = cv2.calcHist([lab], [1, 2], None, [32, 32], [0, 256, 0, 256])
    return cv2.normalize(h, None, alpha=1.0, norm_type=cv2.NORM_L1)


def main():
    data = read("results/experiment_artflow_human/pairs.csv")
    root = Path("downloads/style-transfer-dataset")
    contents = sorted({int(r["content"]) for r in data})
    styles = sorted({int(r["style"]) for r in data})
    cp = {i: root / "contents" / f"content_{i}.jpg" for i in contents}
    sp = {i: root / "styles" / f"style_{i}.jpg" for i in styles}
    images = {str(p): load_input_image(str(p), 512) for p in list(cp.values()) + list(sp.values())}
    histograms = {p: hist(image) for p, image in images.items()}
    structural = {p: image_descriptors(p, 512) for p in images}
    lookup = {}
    for row in data:
        c, s = int(row["content"]), int(row["style"])
        _, _, structural_score = match(structural[str(cp[c])], structural[str(sp[s])])
        if structural_score is None:
            raise ValueError(f"Not enough active patches: content {c}")
        color = 1.0 - cv2.compareHist(histograms[str(cp[c])], histograms[str(sp[s])], cv2.HISTCMP_BHATTACHARYYA)
        lookup[(c, s)] = {"patch": float(row["input_patch_similarity"]), "structure": structural_score,
                          "color": float(color), "rating": float(row["mean_rating"])}
    y = np.array([[lookup[(c, s)]["rating"] for s in styles] for c in contents])
    outrows = []
    for feature in ("patch", "structure", "color"):
        x = np.array([[lookup[(c, s)][feature] for s in styles] for c in contents])
        per_content = [spearman(a, b) for a, b in zip(x, y)]
        per_style = [spearman(a, b) for a, b in zip(x.T, y.T)]
        residual, ci = cluster_bootstrap_interaction(x, y, np.random.default_rng(84), 2000)
        outrows.append({"feature": feature, "content_mean_rho": float(np.mean(per_content)),
                        "content_positive": sum(r > 0 for r in per_content), "content_n": len(per_content),
                        "style_mean_rho": float(np.mean(per_style)), "style_positive": sum(r > 0 for r in per_style),
                        "style_n": len(per_style), "rank_residual_correlation": residual,
                        "residual_ci_low": ci[0], "residual_ci_high": ci[1]})
        print(outrows[-1], flush=True)
    out = Path("results/experiment_artflow_factors")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", outrows)
    write_json(out / "manifest.json", {"status": "post-hoc exploratory", "n": 2500, "method": "rank residual correlation after subtracting image-specific additive means; bootstrap over crossed content/style indices",
        "limits": "Different method (ArtFlow), not StyleID; input images cropped contrary to ArtFlow original; features share low-level statistics, no inference of separate or causal effects, no multiple comparison adjustment."})
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
