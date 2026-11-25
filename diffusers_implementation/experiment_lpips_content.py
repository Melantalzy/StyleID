"""Independent LPIPS-Alex content-distance sensitivity for StyleID on/off.

Uses official TorchVision AlexNet backbone and official LPIPS v0.1 learned
linear weights. Only content perceptual distance is measured, not style quality.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import alexnet

from experiment_topk import spearman, write_csv, write_json
from run_styleid_diffusers import load_input_image


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LPIPSAlex(torch.nn.Module):
    """Equivalent to non-spatial official LPIPS v0.1 in evaluation mode."""
    def __init__(self, backbone_path, linear_path):
        super().__init__()
        self.features = alexnet(weights=None).features
        state = torch.load(backbone_path, map_location="cpu", weights_only=True)
        self.features.load_state_dict({k.removeprefix("features."): v for k, v in state.items() if k.startswith("features.")}, strict=True)
        learned = torch.load(linear_path, map_location="cpu", weights_only=True)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(learned[f"lin{i}.model.1.weight"], requires_grad=False) for i in range(5)])
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])

    def extract(self, images):
        value = (images - self.shift) / self.scale
        features = []
        for i, layer in enumerate(self.features):
            value = layer(value)
            if i in (1, 4, 7, 9, 11):
                features.append(value / (torch.sqrt(torch.sum(value ** 2, dim=1, keepdim=True)) + 1e-10))
        return features

    def distance(self, content_features, output_features):
        result = None
        for weight, a, b in zip(self.weights, content_features, output_features):
            value = F.conv2d((a - b) ** 2, weight).mean(dim=(1, 2, 3))
            result = value if result is None else result + value
        return result


def to_tensor(image):
    array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1


def fetch_groups():
    # All three sets have already been analysed; this is an exploratory metric
    # sensitivity test, not a fresh pre-registered independent test.
    original = read("results/experiment_attention_full/paired_conditions.csv")
    original_inputs = {(r["content"], r["style"]): r for r in read("results/experiment_baseline_gain/paired_baseline_gain.csv")}
    groups = []
    for row in original:
        ref = original_inputs[(row["content"], row["style"])]
        groups.append({"dataset": "original", "content": row["content"], "style": row["style"],
                       "content_path": str(Path("data/cnt") / f"{row['content']}.png"),
                       "input_patch_similarity": float(ref["input_patch_similarity"]),
                       "on_output": row["on_output"], "off_output": row["off_output"]})
    for row in read("results/experiment_unseen_validation/paired_conditions.csv"):
        groups.append({"dataset": "data_vis", "content": row["content"], "style": row["style"],
                       "content_path": str(Path("data_vis/cnt") / f"{row['content']}.png"),
                       "input_patch_similarity": float(row["input_patch_similarity"]),
                       "on_output": row["on_output"], "off_output": row["off_output"]})
    for row in read("results/experiment_artflow_new_styles/paired_conditions.csv"):
        groups.append({"dataset": "artflow_new_styles", "content": row["content"], "style": row["style"],
                       "content_path": str(Path("downloads/style-transfer-dataset/contents") / f"content_{row['content']}.jpg"),
                       "input_patch_similarity": float(row["input_patch_similarity"]),
                       "on_output": row["on_output"], "off_output": row["off_output"]})
    return groups


def main():
    root = Path("downloads/torchvision")
    backbone = root / "alexnet-owt-7be5be79.pth"
    linear = Path("downloads/PerceptualSimilarity/lpips/weights/v0.1/alex.pth")
    if not backbone.is_file() or not linear.is_file():
        raise FileNotFoundError("AlexNet or LPIPS weights are missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experiment")
    model = LPIPSAlex(backbone, linear).eval().requires_grad_(False).to("cuda")
    rows = fetch_groups()
    expected = {"original": 920, "data_vis": 117, "artflow_new_styles": 280}
    for dataset, n in expected.items():
        if sum(r["dataset"] == dataset for r in rows) != n:
            raise ValueError(f"Incomplete {dataset}")
    content_cache = {}
    with torch.inference_mode():
        for path in sorted({r["content_path"] for r in rows}):
            image = load_input_image(path, 512)
            content_cache[path] = model.extract(to_tensor(image).unsqueeze(0).to("cuda"))
        for i, row in enumerate(rows):
            tensors = []
            for field in ("on_output", "off_output"):
                with Image.open(row[field]) as image:
                    tensors.append(to_tensor(image.convert("RGB")))
            output_features = model.extract(torch.stack(tensors).to("cuda"))
            distance = model.distance(content_cache[row["content_path"]], output_features).cpu().numpy()
            row["lpips_on"] = float(distance[0])
            row["lpips_off"] = float(distance[1])
            row["lpips_improvement"] = float(distance[1] - distance[0])
            if (i + 1) % 100 == 0:
                print(f"LPIPS processed {i+1}/{len(rows)}", flush=True)
    summaries = []
    rng = np.random.default_rng(113)
    for dataset in expected:
        subset = [r for r in rows if r["dataset"] == dataset]
        contents = sorted({r["content"] for r in subset})
        styles = sorted({r["style"] for r in subset})
        lookup = {(r["content"], r["style"]): r for r in subset}
        x = np.array([[lookup[(c, s)]["input_patch_similarity"] for s in styles] for c in contents])
        y = np.array([[lookup[(c, s)]["lpips_improvement"] for s in styles] for c in contents])
        rho = [spearman(a, b) for a, b in zip(x, y)]
        boot = []
        for _ in range(3000):
            ci = rng.integers(0, len(contents), len(contents))
            si = rng.integers(0, len(styles), len(styles))
            values = [spearman(x[i, si], y[i, si]) for i in ci]
            values = [v for v in values if v is not None]
            if values:
                boot.append(float(np.mean(values)))
        low, high = np.quantile(boot, [.025, .975])
        summaries.append({"dataset": dataset, "n_contents": len(contents), "n_styles": len(styles),
                          "mean_rho": float(np.mean(rho)), "positive_contents": sum(v > 0 for v in rho),
                          "negative_contents": sum(v < 0 for v in rho), "ci_low": float(low), "ci_high": float(high)})
        print(summaries[-1], flush=True)
    out = Path("results/experiment_lpips_content")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pair_scores.csv", rows)
    write_csv(out / "summary.csv", summaries)
    write_json(out / "manifest.json", {"metric": "LPIPS v0.1 AlexNet, official learned weights; nonspatial; 512 RGB",
        "backbone_sha256": sha256(backbone), "linear_sha256": sha256(linear),
        "device": "cuda", "dtype": "float32", "improvement": "LPIPS(content, off) - LPIPS(content, on); positive is lower distance with injection",
        "limits": "All samples and predictors previously analysed, metric sensitivity exploratory. LPIPS was trained for local perceptual judgments and penalizes intentional stylization; it is not an overall style-transfer quality measure. Pairwise outcomes use JPEGs, whereas original project diagnostics used pre-compression images. Shared styles across contents."})
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
