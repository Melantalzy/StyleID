"""Audit reference-style similarity before transfer and proxy-score gains.

Exploratory: reuses previously analysed data, no untouched confirmation set.
The patch and color baselines intentionally reuse the evaluation metrics to
detect baseline-dependent score effects; they are not independent assessments.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_topk import ranks, spearman, write_csv, write_json
from run_styleid_diffusers import load_input_image
from single_image_metrics import _patch_descriptors, _color_histogram_distance


def paired_patch_similarity(content_descriptor, style_descriptor):
    # Exactly the same directional nearest-neighbour metric as the evaluator.
    distance = np.linalg.norm(content_descriptor[:, None] - style_descriptor[None, :], axis=2)
    return float(np.exp(-distance.min(axis=1).mean()))


def double_demean(matrix):
    return matrix - matrix.mean(axis=0, keepdims=True) - matrix.mean(axis=1, keepdims=True) + matrix.mean()


def corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def cluster_bootstrap_interaction(x, y, rng, count=2000):
    # Rank globally, then remove content and style means. This is an
    # exploratory within-matrix residual association, not partial Spearman.
    xr = ranks(x.ravel()).reshape(x.shape)
    yr = ranks(y.ravel()).reshape(y.shape)
    estimate = corr(double_demean(xr), double_demean(yr))
    samples = []
    for _ in range(count):
        rows = rng.integers(0, x.shape[0], x.shape[0])
        cols = rng.integers(0, x.shape[1], x.shape[1])
        a = xr[np.ix_(rows, cols)]
        b = yr[np.ix_(rows, cols)]
        v = corr(double_demean(a), double_demean(b))
        if v is not None:
            samples.append(v)
    return estimate, [float(v) for v in np.quantile(samples, [.025, .975])] if samples else None


def main():
    source = Path("results/all_default/evaluation_records.csv")
    out = Path("results/experiment_baseline_gain")
    with open(source, encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    contents = {Path(r["content_path"]).stem: Path(r["content_path"]) for r in records}
    styles = {Path(r["style_path"]).stem: Path(r["style_path"]) for r in records}
    if len(records) != len(contents) * len(styles):
        raise ValueError("Incomplete or duplicate input matrix")
    images = {str(p): load_input_image(str(p), 512) for p in list(contents.values()) + list(styles.values())}
    descriptors = {p: _patch_descriptors(im) for p, im in images.items()}
    candidates = []
    for r in records:
        content = Path(r["content_path"]).stem
        style = Path(r["style_path"]).stem
        cpath, spath = str(contents[content]), str(styles[style])
        pre_patch = paired_patch_similarity(descriptors[cpath], descriptors[spath])
        pre_color = 1.0 - _color_histogram_distance(images[cpath], images[spath])
        out_patch = float(r["style_patch_similarity"])
        out_color = float(r["style_color_similarity"])
        candidates.append({"content": content, "style": style,
                           "input_patch_similarity": pre_patch, "output_patch_similarity": out_patch,
                           "patch_gain": out_patch - pre_patch,
                           "input_color_similarity": pre_color, "output_color_similarity": out_color,
                           "color_gain": out_color - pre_color,
                           "content_structure_score": float(r["content_structure_score"]),
                           "content_ssim_gray": float(r["content_ssim_gray"]),
                           "content_edge_f1": float(r["content_edge_f1"])})
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_baseline_gain.csv", candidates)
    lookup = {(r["content"], r["style"]): r for r in candidates}
    cids, sids = sorted(contents), sorted(styles)
    hypotheses = [("input_patch_similarity", "output_patch_similarity"),
                  ("input_patch_similarity", "patch_gain"),
                  ("input_patch_similarity", "content_structure_score"),
                  ("input_patch_similarity", "content_ssim_gray"),
                  ("input_patch_similarity", "content_edge_f1"),
                  ("input_color_similarity", "output_color_similarity"),
                  ("input_color_similarity", "color_gain"),
                  ("input_color_similarity", "content_structure_score"),
                  ("input_color_similarity", "content_ssim_gray"),
                  ("input_color_similarity", "content_edge_f1")]
    rng = np.random.default_rng(42)
    group_rows, summaries = [], []
    for feature, target in hypotheses:
        matrix_x = np.array([[lookup[(c, s)][feature] for s in sids] for c in cids])
        matrix_y = np.array([[lookup[(c, s)][target] for s in sids] for c in cids])
        for group_type, labels, x, y in (("content", cids, matrix_x, matrix_y),
                                       ("style", sids, matrix_x.T, matrix_y.T)):
            coefficients = []
            for label, xx, yy in zip(labels, x, y):
                rho = spearman(xx, yy)
                if rho is not None:
                    coefficients.append(rho)
                group_rows.append({"feature": feature, "target": target, "grouping": group_type,
                                   "group": label, "n": len(xx), "rho": rho})
            vals = np.asarray(coefficients)
            summaries.append({"feature": feature, "target": target, "grouping": group_type,
                              "n_groups": len(vals), "rho_median": float(np.median(vals)),
                              "rho_min": float(np.min(vals)), "rho_max": float(np.max(vals)),
                              "positive": int(np.sum(vals > 0)), "negative": int(np.sum(vals < 0)),
                              "interaction_rho": None, "interaction_ci_low": None, "interaction_ci_high": None})
        value, ci = cluster_bootstrap_interaction(matrix_x, matrix_y, rng)
        summaries.append({"feature": feature, "target": target, "grouping": "rank_double_demeaned",
                          "n_groups": len(cids) * len(sids), "rho_median": None, "rho_min": None,
                          "rho_max": None, "positive": None, "negative": None,
                          "interaction_rho": value, "interaction_ci_low": ci[0] if ci else None,
                          "interaction_ci_high": ci[1] if ci else None})
        print(feature, target, value, ci, flush=True)
    write_csv(out / "correlations_by_group.csv", group_rows)
    write_csv(out / "summary.csv", summaries)
    write_json(out / "manifest.json", {"status": "exploratory, all data previously used", "input_images": len(images),
               "content_count": len(cids), "style_count": len(sids), "pairs": len(candidates),
               "targets": ["final proxy", "final proxy minus input proxy", "content structure proxy"],
               "bootstrap": "2000 independent resamples of content and style indices, two-way rank-residual correlation",
               "limitations": "Input and outcome use identical metric definition; proxy correlation partly tautological. Gain can exhibit mathematical coupling and regression to mean; diagnostic only. Outputs generated earlier without recorded image hashes/model revision. No human validation or independent test set."})
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (feature, target) in zip(axes.flat, [hypotheses[0], hypotheses[1], hypotheses[5], hypotheses[6]]):
        ax.scatter([r[feature] for r in candidates], [r[target] for r in candidates], s=7, alpha=.25)
        ax.set(xlabel=feature, ylabel=target)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "pooled_scatter_exploratory.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {len(candidates)} pairs to {out.resolve()}")


if __name__ == "__main__":
    main()
