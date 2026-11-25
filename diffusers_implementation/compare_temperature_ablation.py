"""Compare paired default StyleID outputs with attention scale 1.0 vs 1.5."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from experiment_topk import write_csv, write_json


METRICS = (
    "content_structure_score", "content_ssim_gray", "content_edge_f1",
    "style_patch_similarity", "style_gram_similarity", "style_color_similarity",
    "balanced_score",
)


def load_records(path, temperature, content):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"ddim_steps": "20", "resolution": "512", "gamma": "0.75",
                "layers": "7,8,9,10,11", "initial_adain": "True",
                "attention_injection": "True", "metric_version": "styleid_vae_pairwise_v1",
                "runtime_device": "cuda", "runtime_precision": "float16"}
    rows = [r for r in rows if Path(r["content_path"]).name == content]
    result = {}
    for row in rows:
        if float(row["temperature"]) != temperature:
            raise ValueError(f"Unexpected temperature: {row['temperature']}")
        for key, value in required.items():
            if row[key] != value:
                raise ValueError(f"Mismatched {key} for {row['run_id']}")
        key = str(Path(row["style_path"]).resolve()).casefold()
        if key in result:
            raise ValueError(f"Duplicate pair: {key}")
        for field in ("style_path", "content_path", "output_path"):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="results/all_default/evaluation_records.csv")
    parser.add_argument("--ablation", default="results/temperature_ablation_t1/evaluation_records.csv")
    parser.add_argument("--save_dir", default="results/temperature_ablation_comparison")
    parser.add_argument("--content", default="00.png")
    parser.add_argument("--expected_styles", type=int, default=46)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    cfg = parser.parse_args()
    if cfg.bootstrap < 1:
        raise ValueError("Bootstrap must be positive")
    baseline = load_records(cfg.baseline, 1.5, cfg.content)
    ablation = load_records(cfg.ablation, 1.0, cfg.content)
    if set(baseline) != set(ablation) or len(baseline) != cfg.expected_styles:
        raise ValueError(f"Pair sets incomplete: baseline={len(baseline)} ablation={len(ablation)}")
    pairs = []
    for key in sorted(baseline):
        base, trial = baseline[key], ablation[key]
        if Path(base["content_path"]).resolve() != Path(trial["content_path"]).resolve():
            raise ValueError(f"Content path mismatch: {key}")
        row = {"style": Path(base["style_path"]).name, "content_path": base["content_path"],
               "style_path": base["style_path"], "baseline_output": base["output_path"],
               "ablation_output": trial["output_path"]}
        for metric in METRICS:
            b, a = float(base[metric]), float(trial[metric])
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"Nonfinite {metric}: {key}")
            row[f"{metric}_t1_5"] = b
            row[f"{metric}_t1_0"] = a
            row[f"{metric}_delta"] = a - b
        pairs.append(row)
    out = Path(cfg.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_results.csv", pairs)
    rng = np.random.default_rng(cfg.seed)
    summaries = []
    for metric in METRICS:
        delta = np.array([p[f"{metric}_delta"] for p in pairs])
        draws = rng.integers(0, len(delta), size=(cfg.bootstrap, len(delta)))
        ci = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
        summaries.append({"metric": metric, "n": len(delta),
                          "mean_t1_5": float(np.mean([p[f"{metric}_t1_5"] for p in pairs])),
                          "mean_t1_0": float(np.mean([p[f"{metric}_t1_0"] for p in pairs])),
                          "mean_delta_t1_minus_t1_5": float(np.mean(delta)),
                          "median_delta": float(np.median(delta)),
                          "ci_mean_delta_low": float(ci[0]), "ci_mean_delta_high": float(ci[1]),
                          "positive_count": int(np.sum(delta > 0)), "negative_count": int(np.sum(delta < 0)),
                          "zero_count": int(np.sum(delta == 0))})
    write_csv(out / "summary.csv", summaries)
    write_json(out / "manifest.json", {"parameters": vars(cfg), "difference": "T=1.0 minus T=1.5",
        "method": "Paired comparison; 95% percentile bootstrap over styles, conditional on content 00 and sampled references.",
        "limitations": "Existing baseline records do not include model revision, input hashes, code revision, or environment versions; identical historical provenance cannot be verified. Metrics are unvalidated quality proxies and generated before JPEG compression. This examines full multi-step temperature intervention, not a single mechanism or attention map."})
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, metric in zip(axes.flat, METRICS):
        b = np.array([p[f"{metric}_t1_5"] for p in pairs])
        a = np.array([p[f"{metric}_t1_0"] for p in pairs])
        ax.scatter(b, a, s=20, alpha=0.75)
        lo, hi = min(b.min(), a.min()), max(b.max(), a.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set(title=metric, xlabel="T=1.5 (default)", ylabel="T=1.0")
        ax.grid(alpha=0.2)
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "paired_scatter.png", dpi=160)
    plt.close(fig)
    # Predefined first eight styles, avoiding selection by performance.
    selected = pairs[:8]
    tile, header = 192, 26
    sheet = Image.new("RGB", (4 * tile, (tile + header) * len(selected) + header), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(("Content", "Style", "Default T=1.5", "No scaling T=1.0")):
        draw.text((col * tile + 4, 5), label, fill="black")
    for i, row in enumerate(selected):
        y = header + i * (tile + header)
        draw.text((4, y + 4), row["style"], fill="black")
        for col, field in enumerate(("content_path", "style_path", "baseline_output", "ablation_output")):
            with Image.open(row[field]) as source:
                image = source.convert("RGB")
                image.thumbnail((tile, tile))
                sheet.paste(image, (col * tile + (tile - image.width) // 2, y + header + (tile - image.height) // 2))
    sheet.save(out / "first_8_visual_comparison.jpg", quality=90)
    print(f"Compared {len(pairs)} pairs; results: {out.resolve()}")
    for row in summaries:
        print(f"{row['metric']}: delta={row['mean_delta_t1_minus_t1_5']:.6f}, 95% CI [{row['ci_mean_delta_low']:.6f}, {row['ci_mean_delta_high']:.6f}], positive/negative={row['positive_count']}/{row['negative_count']}")


if __name__ == "__main__":
    main()
