"""Compare reference-style mean grayscale gradient to existing StyleID scores.

Run from the repository root. No images are generated and no model is loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_topk import correlation_stats, write_csv, write_json


PRIMARY_METRICS = ("style_patch_similarity", "content_structure_score")
EXPLORATORY_METRICS = ("style_gram_similarity", "style_color_similarity", "content_edge_f1", "balanced_score")
ALL_METRICS = PRIMARY_METRICS + EXPLORATORY_METRICS


def style_gradient(path, resolution):
    # Same BGR read, central square crop, area resize and grayscale conversion
    # as StyleID's input loader and existing image diagnostics.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")
    height, width = image.shape[:2]
    size = min(height, width)
    top, left = (height - size) // 2, (width - size) // 2
    image = cv2.resize(image[top:top + size, left:left + size],
                       (resolution, resolution), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # scale=1/8 makes a unit slope in intensity per pixel produce a unit
    # derivative for the 3x3 Sobel kernel, away from image boundaries.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REFLECT_101)
    return float(np.hypot(gx, gy).mean(dtype=np.float64))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_adjust(rows):
    ordered = sorted((r for r in rows if r["family"] == "primary" and r["p"] is not None),
                     key=lambda r: r["p"])
    adjusted = 1.0
    for index in range(len(ordered) - 1, -1, -1):
        adjusted = min(adjusted, ordered[index]["p"] * len(ordered) / (index + 1))
        ordered[index]["bh_q_primary_family"] = adjusted


def plot_scatter_grid(output, pairs, contents, metric):
    columns = 5
    rows = math.ceil(len(contents) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 2.8 * rows), squeeze=False)
    for ax, content in zip(axes.flat, contents):
        points = [p for p in pairs if p["content"] == content]
        ax.scatter([p["style_gradient"] for p in points], [p[metric] for p in points], s=12, alpha=0.8)
        ax.set_title(f"Content {content}")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(contents):]:
        ax.set_visible(False)
    fig.supxlabel("Style mean grayscale Sobel gradient")
    fig.supylabel(metric)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="results/all_default/evaluation_records.csv")
    parser.add_argument("--style_dir", default="data/sty")
    parser.add_argument("--save_dir", default="results/experiment_style_gradient")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    cfg = parser.parse_args()
    if cfg.resolution <= 0 or cfg.bootstrap <= 0 or cfg.permutations <= 0:
        raise ValueError("Resolution and resampling counts must be positive")
    with open(cfg.records, newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("Evaluation records are empty")
    expected = {"ddim_steps": "20", "resolution": str(cfg.resolution), "gamma": "0.75",
                "temperature": "1.5", "layers": "7,8,9,10,11", "initial_adain": "True",
                "attention_injection": "True", "metric_version": "styleid_vae_pairwise_v1",
                "runtime_device": "cuda", "runtime_precision": "float16"}
    paths = sorted((p.resolve() for p in Path(cfg.style_dir).iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}),
                   key=lambda p: p.name)
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Style images must be nonempty and have unique stems")
    style_values = {str(p).casefold(): style_gradient(p, cfg.resolution) for p in paths}
    style_rows = [{"style": p.stem, "style_path": str(p), "sha256": sha256(p),
                   "style_gradient": style_values[str(p).casefold()]} for p in paths]
    pairs = []
    seen = set()
    for record in records:
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                raise ValueError(f"Unexpected {field}={record.get(field)!r} for run {record.get('run_id')}")
        style_path = str(Path(record["style_path"]).resolve()).casefold()
        if style_path not in style_values:
            raise ValueError(f"Style path missing from inputs: {record['style_path']}")
        content = Path(record["content_path"]).stem
        style = Path(record["style_path"]).stem
        pair_id = (content, style)
        if pair_id in seen:
            raise ValueError(f"Duplicate content/style pair: {pair_id}")
        seen.add(pair_id)
        if not Path(record["content_path"]).is_file() or not Path(record["output_path"]).is_file():
            raise FileNotFoundError(f"Missing content or output for {pair_id}")
        values = {key: float(record[key]) for key in ALL_METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite metric for {pair_id}")
        pairs.append({"content": content, "style": style, "style_gradient": style_values[style_path],
                      **values, "run_id": record["run_id"], "output_path": record["output_path"]})
    contents = sorted({p["content"] for p in pairs})
    styles = {p.stem for p in paths}
    for content in contents:
        observed = {p["style"] for p in pairs if p["content"] == content}
        if observed != styles:
            raise ValueError(f"Incomplete style set for content {content}; missing {sorted(styles - observed)}")
    out = Path(cfg.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "style_gradients.csv", style_rows)
    write_csv(out / "joined_pairs.csv", pairs)
    write_json(out / "manifest.json", {"parameters": vars(cfg), "record_sha256": sha256(cfg.records),
        "style_count": len(styles), "content_count": len(contents), "pair_count": len(pairs),
        "definition": "mean hypot of 3x3 Sobel x/y (scale=1/8) on central-square-cropped, area-resized, OpenCV grayscale [0,1]; BORDER_REFLECT_101",
        "primary_metrics": PRIMARY_METRICS, "exploratory_metrics": EXPLORATORY_METRICS,
        "limitations": "Historical output records contain no input hashes or model revision; matching original inputs and provenance cannot be guaranteed. Fixed styles reused across contents; no pooled independence claim."})
    correlations = []
    for content in contents:
        subset = [p for p in pairs if p["content"] == content]
        x = [p["style_gradient"] for p in subset]
        for metric in ALL_METRICS:
            result = correlation_stats(x, [p[metric] for p in subset], cfg.bootstrap, cfg.permutations, cfg.seed)
            ci = result["bootstrap_ci_95"]
            correlations.append({"content": content, "metric": metric,
                "family": "primary" if metric in PRIMARY_METRICS else "exploratory",
                "n_styles": result["n"], "rho": result["spearman_rho"],
                "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
                "p": result["permutation_p_two_sided"], "bh_q_primary_family": None,
                "bootstrap_valid": result["bootstrap_valid"]})
        print(f"Analysed content {content} ({len(subset)} styles)", flush=True)
    bh_adjust(correlations)
    write_csv(out / "correlations_by_content.csv", correlations)
    by_metric = []
    for metric in ALL_METRICS:
        selected = [r for r in correlations if r["metric"] == metric and r["rho"] is not None]
        if not selected:
            continue
        rho = np.asarray([r["rho"] for r in selected])
        by_metric.append({"metric": metric, "family": selected[0]["family"],
                          "n_contents": len(selected), "rho_median": float(np.median(rho)),
                          "rho_min": float(np.min(rho)), "rho_max": float(np.max(rho)),
                          "positive_count": int(np.sum(rho > 0)), "negative_count": int(np.sum(rho < 0)),
                          "q_below_0_05_count": sum(r["bh_q_primary_family"] is not None and r["bh_q_primary_family"] < 0.05 for r in selected) if selected[0]["family"] == "primary" else None})
    write_csv(out / "correlation_summary.csv", by_metric)
    for metric in PRIMARY_METRICS:
        plot_scatter_grid(out / f"{metric}_scatter_by_content.png", pairs, contents, metric)
    print(json.dumps(by_metric, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
