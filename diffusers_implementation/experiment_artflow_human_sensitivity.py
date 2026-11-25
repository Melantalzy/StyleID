"""Post-hoc sensitivity across ArtFlow style sizes and individual raters."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from experiment_topk import spearman, write_csv, write_json


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    inputs = {(int(r["content"]), int(r["style"])): float(r["input_patch_similarity"])
              for r in read("results/experiment_artflow_human/pairs.csv")}
    pattern = re.compile(r"^content_(\d+)___style_(\d+)___(150|300|500|700)\.jpg$")
    data = {}
    for row in read("downloads/style-transfer-dataset/user-ratings-all.csv"):
        m = pattern.match(row["name"])
        if m:
            key = tuple(map(int, m.groups()))
            if key in data:
                raise ValueError(f"Duplicate rating: {key}")
            data[key] = row
    contents = sorted({c for c, _ in inputs})
    styles = sorted({s for _, s in inputs})
    x = np.array([[inputs[(c, s)] for s in styles] for c in contents])
    rng = np.random.default_rng(485)
    summary = []
    for size, rating in [(150, "mean_rating"), (300, "mean_rating"), (500, "mean_rating"),
                         (700, "mean_rating"), (500, "1_rating"), (500, "2_rating"), (500, "3_rating")]:
        y = np.array([[float(data[(c, s, size)][rating]) for s in styles] for c in contents])
        rhos = [spearman(a, b) for a, b in zip(x, y)]
        estimates = []
        for _ in range(1000):
            ci = rng.integers(0, len(contents), len(contents))
            si = rng.integers(0, len(styles), len(styles))
            values = [spearman(x[i, si], y[i, si]) for i in ci]
            estimates.append(float(np.mean([v for v in values if v is not None])))
        low, high = np.quantile(estimates, [.025, .975])
        summary.append({"style_size": size, "rating": rating, "mean_rho": float(np.mean(rhos)),
                        "rho_min": float(np.min(rhos)), "rho_max": float(np.max(rhos)),
                        "positive_contents": sum(r > 0 for r in rhos), "n_contents": len(rhos),
                        "ci_low": float(low), "ci_high": float(high)})
        print(summary[-1], flush=True)
    out = Path("results/experiment_artflow_human_sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", summary)
    write_json(out / "manifest.json", {"status": "post-hoc; the same 50 contents and styles appear in each condition",
               "predictor": "fixed StyleID-preprocessed descriptor, does not incorporate the 150/300/500/700 generation sizes",
               "bootstrap": "1000 crossed resamples of contents and styles per condition",
               "limitations": "ArtFlow and human ratings cannot serve as StyleID labels. Outputs recolored; multiple comparisons exploratory; no new independent images."})
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
