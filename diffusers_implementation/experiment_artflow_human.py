"""Test frozen input patch similarity against ArtFlow human scores.

Data: Kitov et al., style-transfer-dataset, CC BY 4.0. This is not a StyleID
experiment: results use ArtFlow, color-recolored outputs and 500 style size.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from experiment_baseline_gain import cluster_bootstrap_interaction, paired_patch_similarity
from experiment_style_gradient import sha256
from experiment_topk import spearman, write_csv, write_json
from run_styleid_diffusers import load_input_image
from single_image_metrics import _patch_descriptors


def main():
    root = Path("downloads/style-transfer-dataset")
    with open(root / "user-ratings-all.csv", encoding="utf-8-sig", newline="") as stream:
        ratings = list(csv.DictReader(stream))
    selected = []
    pattern = re.compile(r"^content_(\d+)___style_(\d+)___500\.jpg$")
    for row in ratings:
        found = pattern.match(row["name"])
        if found:
            selected.append((found.group(1), found.group(2), row))
    if len(selected) != 2500 or len({(c, s) for c, s, _ in selected}) != 2500:
        raise ValueError(f"Expected 2500 unique pairs at style size 500, got {len(selected)}")
    content_ids = sorted({int(c) for c, _, _ in selected})
    style_ids = sorted({int(s) for _, s, _ in selected})
    if len(content_ids) != 50 or len(style_ids) != 50:
        raise ValueError("Expected 50 distinct contents and 50 styles")
    contents = {str(c): root / "contents" / f"content_{c}.jpg" for c in content_ids}
    styles = {str(s): root / "styles" / f"style_{s}.jpg" for s in style_ids}
    print("Extracting fixed descriptors from 100 input images", flush=True)
    descriptors = {str(p): _patch_descriptors(load_input_image(str(p), 512))
                   for p in list(contents.values()) + list(styles.values())}
    rows = []
    for c, s, row in selected:
        raw = [float(row[f"{i}_rating"]) for i in (1, 2, 3)]
        mean = float(row["mean_rating"])
        if not np.isclose(mean, np.mean(raw)):
            raise ValueError(f"Mean mismatch {row['name']}")
        rows.append({"content": c, "style": s, "style_size": 500,
                     "input_patch_similarity": paired_patch_similarity(descriptors[str(contents[c])], descriptors[str(styles[s])]),
                     "rating_1": raw[0], "rating_2": raw[1], "rating_3": raw[2], "mean_rating": mean})
    lookup = {(int(r["content"]), int(r["style"])): r for r in rows}
    x = np.array([[lookup[(c, s)]["input_patch_similarity"] for s in style_ids] for c in content_ids])
    y = np.array([[lookup[(c, s)]["mean_rating"] for s in style_ids] for c in content_ids])
    by_group = []
    for label, ids, a, b in (("content", content_ids, x, y), ("style", style_ids, x.T, y.T)):
        for ident, xx, yy in zip(ids, a, b):
            by_group.append({"grouping": label, "group": ident, "n": len(xx), "rho": spearman(xx, yy)})
    group = [r for r in by_group if r["grouping"] == "content" and r["rho"] is not None]
    coefficients = [r["rho"] for r in group]
    rng = np.random.default_rng(90210)
    estimates = []
    for _ in range(2000):
        ci = rng.integers(0, 50, 50)
        si = rng.integers(0, 50, 50)
        rhos = [spearman(x[i, si], y[i, si]) for i in ci]
        estimates.append(float(np.mean([r for r in rhos if r is not None])))
    low, high = np.quantile(estimates, [.025, .975])
    interaction, interaction_ci = cluster_bootstrap_interaction(x, y, np.random.default_rng(90211), 2000)
    out = Path("results/experiment_artflow_human")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pairs.csv", rows)
    write_csv(out / "correlations_by_group.csv", by_group)
    summary = {"algorithm": "ArtFlow, color-recolored (paper procedure), NOT StyleID",
               "style_size": 500, "content_count": 50, "style_count": 50, "pair_count": len(rows),
               "mean_rho_by_content": float(np.mean(coefficients)),
               "median_rho_by_content": float(np.median(coefficients)),
               "min_rho": float(np.min(coefficients)), "max_rho": float(np.max(coefficients)),
               "positive": sum(v > 0 for v in coefficients), "negative": sum(v < 0 for v in coefficients),
               "crossed_bootstrap_95_ci": [float(low), float(high)], "bootstrap_iterations": 2000,
               "two_way_rank_residual_correlation": interaction,
               "two_way_rank_residual_bootstrap_95_ci": interaction_ci,
               "ratings_file_sha256": sha256(root / "user-ratings-all.csv"),
               "limitations": "Styles and contents here have not been proven independent of earlier datasets. Input descriptor uses StyleID center crop/512 preprocessing unlike ArtFlow's original sizes; evaluator ratings are ArtFlow color-recolored results, not StyleID or attention injection. Correlations are observational. Human labels subjectivity, 3 raters."}
    write_json(out / "summary.json", summary)
    print(summary, flush=True)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
