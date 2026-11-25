"""Exploratory AlexNet Gram style similarity on three existing datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from experiment_lpips_content import LPIPSAlex, fetch_groups, sha256, to_tensor
from experiment_topk import spearman, write_csv, write_json
from run_styleid_diffusers import load_input_image


def gram_features(model, tensor):
    # Feature statistics, not LPIPS and not the canonical VGG Gram metric.
    x = (tensor - model.shift) / model.scale
    grams = []
    for i, layer in enumerate(model.features):
        x = layer(x)
        if i in (1, 4, 7, 9, 11):
            v = x.flatten(2)
            gram = torch.bmm(v, v.transpose(1, 2)) / v.shape[-1]
            denom = torch.linalg.vector_norm(gram, dim=(1, 2), keepdim=True).clamp_min(1e-8)
            grams.append(gram / denom)
    return grams


def similarity(reference, generated):
    distances = [(torch.linalg.vector_norm(a - b, dim=(1, 2)) / (2.0 ** .5)).clamp(0, 1)
                 for a, b in zip(reference, generated)]
    return 1.0 - torch.stack(distances).mean(dim=0)


def main():
    backbone = Path("downloads/torchvision/alexnet-owt-7be5be79.pth")
    linear = Path("downloads/PerceptualSimilarity/lpips/weights/v0.1/alex.pth")
    if not torch.cuda.is_available() or not backbone.is_file() or not linear.is_file():
        raise RuntimeError("CUDA and AlexNet weights are required")
    model = LPIPSAlex(backbone, linear).eval().requires_grad_(False).to("cuda")
    rows = fetch_groups()
    def style_path(r):
        if r["dataset"] == "original":
            return Path("data/sty") / f"{r['style']}.png"
        if r["dataset"] == "data_vis":
            return Path("data_vis/sty") / f"{r['style']}.png"
        return Path("downloads/style-transfer-dataset/styles") / f"style_{r['style']}.jpg"
    with torch.inference_mode():
        cache = {}
        for path in sorted({str(style_path(r)) for r in rows}):
            img = load_input_image(path, 512)
            cache[path] = gram_features(model, to_tensor(img).unsqueeze(0).to("cuda"))
        for i, row in enumerate(rows):
            images = []
            for field in ("on_output", "off_output"):
                with Image.open(row[field]) as img:
                    images.append(to_tensor(img.convert("RGB")))
            generated = gram_features(model, torch.stack(images).to("cuda"))
            scores = similarity(cache[str(style_path(row))], generated).cpu().numpy()
            row["alex_gram_on"] = float(scores[0])
            row["alex_gram_off"] = float(scores[1])
            row["alex_gram_delta"] = float(scores[0] - scores[1])
            if (i+1) % 100 == 0:
                print(f"AlexNet Gram {i+1}/{len(rows)}", flush=True)
    rng = np.random.default_rng(291)
    summary = []
    for dataset in ("original", "data_vis", "artflow_new_styles"):
        selected = [r for r in rows if r["dataset"] == dataset]
        contents, styles = sorted({r["content"] for r in selected}), sorted({r["style"] for r in selected})
        table = {(r["content"], r["style"]): r for r in selected}
        x = np.array([[table[(c, s)]["input_patch_similarity"] for s in styles] for c in contents])
        for target in ("alex_gram_on", "alex_gram_delta"):
            y = np.array([[table[(c, s)][target] for s in styles] for c in contents])
            rhos = [spearman(a, b) for a, b in zip(x, y)]
            boot = []
            for _ in range(3000):
                ci = rng.integers(0, len(contents), len(contents))
                si = rng.integers(0, len(styles), len(styles))
                vals = [spearman(x[i, si], y[i, si]) for i in ci]
                vals = [v for v in vals if v is not None]
                if vals:
                    boot.append(float(np.mean(vals)))
            low, high = np.quantile(boot, [.025, .975])
            entry = {"dataset": dataset, "target": target, "mean_rho": float(np.mean(rhos)),
                     "positive": sum(r > 0 for r in rhos), "negative": sum(r < 0 for r in rhos),
                     "ci_low": float(low), "ci_high": float(high), "contents": len(contents), "styles": len(styles)}
            summary.append(entry)
            print(entry, flush=True)
    out = Path("results/experiment_alexnet_gram")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pair_scores.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_json(out / "manifest.json", {"metric": "five-layer equal-weighted AlexNet normalized Gram similarity; higher is closer to style",
        "backbone_sha256": sha256(backbone), "device": "cuda", "dtype": "float32",
        "limits": "Not canonical VGG Gram or validated perceptual quality, reflects global feature statistics. JPEG outputs, 512 center-cropped style references. Post-hoc exploratory, samples previously analysed, no multiplicity correction, styles reused across contents."})
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
