"""Input-only local grayscale structure matching vs existing StyleID scores.

This is an exploratory hand-crafted descriptor, not StyleID's Q/K matching or
a validated perceptual quality measure. Run from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_style_gradient import sha256
from experiment_topk import correlation_stats, write_csv, write_json


METRICS = ("style_patch_similarity", "content_structure_score")
GRID = 8
BINS = 8
TOP_K = 3
ACTIVITY_SCALE = 0.05
ACTIVE_THRESHOLD = 0.10
MIN_ACTIVE = 4


def descriptors_from_gray(gray):
    """Return 64 unsigned orientation distributions and bounded activities."""
    if gray.ndim != 2 or min(gray.shape) < GRID or any(n % GRID for n in gray.shape):
        raise ValueError("Grayscale image dimensions must be divisible by 8")
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REFLECT_101)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    ph, pw = gray.shape[0] // GRID, gray.shape[1] // GRID
    orientations, activities = [], []
    for row in range(GRID):
        for col in range(GRID):
            box = (slice(row * ph, (row + 1) * ph), slice(col * pw, (col + 1) * pw))
            m = magnitude[box]
            hist = np.histogram(angle[box], bins=BINS, range=(0, np.pi), weights=m)[0].astype(np.float64)
            total = hist.sum()
            if total > 1e-12:
                hist /= total
            else:
                hist[:] = 1.0 / BINS
            energy = float(np.mean(m, dtype=np.float64))
            orientations.append(hist)
            activities.append(energy / (energy + ACTIVITY_SCALE))
    return np.asarray(orientations), np.asarray(activities)


def image_descriptors(path, resolution):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    height, width = img.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    img = cv2.resize(img[top:top + side, left:left + side], (resolution, resolution),
                     interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return descriptors_from_gray(gray)


def match(content, style):
    """Match content regions to top-3 style regions, no position alignment."""
    cp, ca = content
    sp, sa = style
    hellinger = np.linalg.norm(np.sqrt(cp[:, None, :]) - np.sqrt(sp[None, :, :]), axis=-1) / math.sqrt(2)
    # Orientation is uninformative for weak-gradient regions; suppress its
    # contribution when either patch has little activity.
    distance = 0.5 * np.abs(ca[:, None] - sa[None, :]) + 0.5 * np.minimum(ca[:, None], sa[None, :]) * hellinger
    similarities = 1.0 - distance
    top = np.partition(similarities, -TOP_K, axis=1)[:, -TOP_K:]
    per_patch = top.mean(axis=1)
    active = ca >= ACTIVE_THRESHOLD
    active_score = float(np.mean(per_patch[active])) if int(active.sum()) >= MIN_ACTIVE else None
    return per_patch, active, active_score


def bh_adjust(rows):
    valid = sorted((r for r in rows if r["p"] is not None), key=lambda r: r["p"])
    value = 1.0
    for index in range(len(valid) - 1, -1, -1):
        value = min(value, valid[index]["p"] * len(valid) / (index + 1))
        valid[index]["q_bh_all_tests"] = value


def scatter(output, pairs, contents, metric):
    ncols = 5
    nrows = math.ceil(len(contents) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 2.8), squeeze=False)
    for ax, content in zip(axes.flat, contents):
        group = [r for r in pairs if r["content"] == content and r["match_active"] is not None]
        ax.scatter([r["match_active"] for r in group], [r[metric] for r in group], s=12)
        ax.set_title(f"Content {content}")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(contents):]:
        ax.set_visible(False)
    fig.supxlabel("Active content patches: local structure match")
    fig.supylabel(metric)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="results/all_default/evaluation_records.csv")
    parser.add_argument("--save_dir", default="results/experiment_local_matching")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    cfg = parser.parse_args()
    if cfg.resolution <= 0 or cfg.resolution % GRID or cfg.bootstrap <= 0 or cfg.permutations <= 0:
        raise ValueError("Resolution must be divisible by 8; resampling counts must be positive")
    with open(cfg.records, encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("No records")
    expected = {"ddim_steps": "20", "resolution": str(cfg.resolution), "gamma": "0.75",
                "temperature": "1.5", "layers": "7,8,9,10,11", "initial_adain": "True",
                "attention_injection": "True", "metric_version": "styleid_vae_pairwise_v1",
                "runtime_device": "cuda", "runtime_precision": "float16"}
    for row in records:
        for name, value in expected.items():
            if row.get(name) != value:
                raise ValueError(f"Mismatch {name} in record {row['run_id']}")
    def unique_paths(field):
        paths = {str(Path(r[field]).resolve()).casefold(): Path(r[field]).resolve() for r in records}
        if len({p.stem for p in paths.values()}) != len(paths):
            raise ValueError(f"Nonunique image stems in {field}")
        return paths
    content_paths = unique_paths("content_path")
    style_paths = unique_paths("style_path")
    print(f"Extracting descriptors for {len(content_paths)} contents and {len(style_paths)} styles", flush=True)
    features = {key: image_descriptors(path, cfg.resolution) for group in (content_paths, style_paths)
                for key, path in group.items()}
    pairs, details, seen = [], [], set()
    for row in records:
        cpath = str(Path(row["content_path"]).resolve()).casefold()
        spath = str(Path(row["style_path"]).resolve()).casefold()
        cid, sid = content_paths[cpath].stem, style_paths[spath].stem
        if (cid, sid) in seen:
            raise ValueError(f"Duplicate pair {(cid, sid)}")
        seen.add((cid, sid))
        if not Path(row["output_path"]).is_file():
            raise FileNotFoundError(row["output_path"])
        values = {metric: float(row[metric]) for metric in METRICS}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Nonfinite metric for {(cid, sid)}")
        matches, active, mean_active = match(features[cpath], features[spath])
        pairs.append({"content": cid, "style": sid, "match_active": mean_active,
                      "match_all": float(matches.mean()), "match_q10": float(np.quantile(matches[active], .1)) if active.sum() >= MIN_ACTIVE else None,
                      "active_count": int(active.sum()), **values, "run_id": row["run_id"],
                      "output_path": row["output_path"]})
        details.extend({"content": cid, "style": sid, "patch": i, "row": i // GRID, "column": i % GRID,
                        "activity": float(features[cpath][1][i]), "active": bool(active[i]), "match_top3": float(value)}
                       for i, value in enumerate(matches))
    if len(seen) != len(content_paths) * len(style_paths):
        raise ValueError("Input records do not form a complete content/style matrix")
    out = Path(cfg.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pair_metrics.csv", pairs)
    write_csv(out / "patch_metrics.csv", details)
    correlations = []
    for grouping in ("content", "style"):
        groups = sorted({r[grouping] for r in pairs})
        for group in groups:
            sample = [r for r in pairs if r[grouping] == group and r["match_active"] is not None]
            for metric in METRICS:
                result = correlation_stats([r["match_active"] for r in sample], [r[metric] for r in sample],
                                           cfg.bootstrap, cfg.permutations, cfg.seed)
                ci = result["bootstrap_ci_95"]
                correlations.append({"grouping": grouping, "group": group, "metric": metric,
                    "n": result["n"], "rho": result["spearman_rho"], "ci_low": ci[0] if ci else None,
                    "ci_high": ci[1] if ci else None, "p": result["permutation_p_two_sided"],
                    "q_bh_all_tests": None, "bootstrap_valid": result["bootstrap_valid"]})
            print(f"Analysed {grouping} {group}: {len(sample)} pairs", flush=True)
    bh_adjust(correlations)
    write_csv(out / "correlations.csv", correlations)
    summary = []
    for grouping in ("content", "style"):
        for metric in METRICS:
            selected = [r for r in correlations if r["grouping"] == grouping and r["metric"] == metric and r["rho"] is not None]
            if not selected:
                continue
            rho = np.asarray([r["rho"] for r in selected])
            summary.append({"grouping": grouping, "metric": metric, "groups": len(selected),
                            "median_rho": float(np.median(rho)), "min_rho": float(np.min(rho)),
                            "max_rho": float(np.max(rho)), "positive": int(np.sum(rho > 0)),
                            "negative": int(np.sum(rho < 0)),
                            "q_lt_0_05": sum(r["q_bh_all_tests"] < 0.05 for r in selected)})
    write_csv(out / "correlation_summary.csv", summary)
    write_json(out / "manifest.json", {"version": "local_hog_match_v1", "parameters": vars(cfg),
        "record_sha256": sha256(cfg.records), "inputs_sha256": {str(p): sha256(p) for d in (content_paths, style_paths) for p in d.values()},
        "content_count": len(content_paths), "style_count": len(style_paths), "pairs": len(pairs),
        "grid": GRID, "bins": BINS, "top_k": TOP_K, "activity_scale": ACTIVITY_SCALE,
        "active_threshold": ACTIVE_THRESHOLD, "min_active": MIN_ACTIVE,
        "distance": "0.5*abs(activity_c-activity_s)+0.5*min(activity_c,activity_s)*Hellinger(unsigned orientation histograms)",
        "primary_score": "mean top-3 similarity across active content patches; undefined if fewer than 4",
        "multiplicity": "BH across both metrics and both grouping directions; same images reused, tests dependent",
        "limitations": "Hand-crafted grayscale local statistics; arbitrary uncalibrated thresholds; overlap with patch evaluation; no inference of style correctness, Q/K or causation; historical source hashes/model revision unavailable."})
    contents = sorted({p["content"] for p in pairs})
    for metric in METRICS:
        scatter(out / f"{metric}_by_content.png", pairs, contents, metric)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
