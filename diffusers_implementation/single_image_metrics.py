"""Per-image diagnostics for StyleID experiments.

The metrics in this module are deliberately pairwise: they compare one
stylized image with its content and style references.  FID and ArtFID are
distribution metrics and therefore do not belong in a per-run experiment
record.

Deep feature metrics reuse the Stable Diffusion VAE that is already loaded by
StyleID.  This keeps the Windows/offline launcher self-contained and avoids a
second large evaluation model download.
"""

from __future__ import annotations

import csv
import json
import math
import os
import uuid
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F


METRIC_FIELDS = (
    "content_structure_distance",
    "content_structure_score",
    "content_ssim_gray",
    "content_edge_f1",
    "style_gram_distance",
    "style_gram_similarity",
    "style_color_distance",
    "style_color_similarity",
    "style_patch_distance",
    "style_patch_similarity",
    "style_patch_coverage",
    "content_score",
    "style_score",
    "balanced_score",
)


def _as_uint8_rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape H x W x 3")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).round().astype(np.uint8)
    return image


def _vae_device_dtype(vae):
    parameter = next(vae.parameters())
    return parameter.device, parameter.dtype


def _extract_vae_features(image, vae, pool_size=32):
    """Collect compact multi-scale encoder features without retaining graphs."""
    features = []
    hooks = []

    def capture_feature(_module, _inputs, output):
        feature = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(feature) or feature.ndim != 4:
            return
        height = min(pool_size, feature.shape[-2])
        width = min(pool_size, feature.shape[-1])
        pooled = F.adaptive_avg_pool2d(feature.detach().float(), (height, width))
        features.append(pooled.cpu())

    # Skip the first, very high-resolution block.  The remaining blocks retain
    # useful texture and layout information at a modest memory cost.
    down_blocks = list(vae.encoder.down_blocks)
    selected_blocks = down_blocks[1:] if len(down_blocks) > 1 else down_blocks
    for block in selected_blocks:
        hooks.append(block.register_forward_hook(capture_feature))

    device, dtype = _vae_device_dtype(vae)
    rgb = _as_uint8_rgb(image)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype) / 127.5 - 1.0

    try:
        with torch.no_grad():
            vae.encode(tensor).latent_dist.mode()
    finally:
        for hook in hooks:
            hook.remove()

    if not features:
        raise RuntimeError("No VAE encoder features were captured for evaluation")
    return features


def _self_similarity_distance(first, second, spatial_size=16):
    first = F.adaptive_avg_pool2d(first.float(), (spatial_size, spatial_size))
    second = F.adaptive_avg_pool2d(second.float(), (spatial_size, spatial_size))

    def similarity(feature):
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = F.normalize(tokens, dim=-1, eps=1e-8)
        return torch.bmm(tokens, tokens.transpose(1, 2))

    # Cosine similarities lie in [-1, 1], so division by two maps the largest
    # possible absolute discrepancy to one.
    distance = (similarity(first) - similarity(second)).abs().mean() / 2.0
    return float(distance.clamp(0.0, 1.0).item())


def _normalized_gram(feature):
    values = feature.float().flatten(2)
    gram = torch.bmm(values, values.transpose(1, 2)) / values.shape[-1]
    return gram / torch.linalg.vector_norm(gram, dim=(1, 2), keepdim=True).clamp_min(1e-8)


def _multi_scale_gram_distance(first_features, second_features):
    distances = []
    for first, second in zip(first_features, second_features):
        delta = _normalized_gram(first) - _normalized_gram(second)
        # The maximum Frobenius distance between unit-normalized matrices is
        # sqrt(2); report an approximately [0, 1] distance.
        distance = torch.linalg.vector_norm(delta, dim=(1, 2)).mean() / math.sqrt(2.0)
        distances.append(float(distance.clamp(0.0, 1.0).item()))
    return float(np.mean(distances))


def _gray_ssim(first, second):
    first_gray = cv2.cvtColor(_as_uint8_rgb(first), cv2.COLOR_RGB2GRAY).astype(np.float64)
    second_gray = cv2.cvtColor(_as_uint8_rgb(second), cv2.COLOR_RGB2GRAY).astype(np.float64)
    if first_gray.shape != second_gray.shape:
        second_gray = cv2.resize(
            second_gray,
            (first_gray.shape[1], first_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    kernel = (11, 11)
    sigma = 1.5
    mu_first = cv2.GaussianBlur(first_gray, kernel, sigma)
    mu_second = cv2.GaussianBlur(second_gray, kernel, sigma)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_product = mu_first * mu_second

    sigma_first_sq = cv2.GaussianBlur(first_gray * first_gray, kernel, sigma) - mu_first_sq
    sigma_second_sq = cv2.GaussianBlur(second_gray * second_gray, kernel, sigma) - mu_second_sq
    sigma_product = cv2.GaussianBlur(first_gray * second_gray, kernel, sigma) - mu_product

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mu_product + c1) * (2.0 * sigma_product + c2)
    denominator = (mu_first_sq + mu_second_sq + c1) * (
        sigma_first_sq + sigma_second_sq + c2
    )
    score = np.mean(numerator / np.maximum(denominator, 1e-12))
    return float(np.clip(score, 0.0, 1.0))


def _edge_f1(first, second):
    first_gray = cv2.cvtColor(_as_uint8_rgb(first), cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(_as_uint8_rgb(second), cv2.COLOR_RGB2GRAY)
    if first_gray.shape != second_gray.shape:
        second_gray = cv2.resize(
            second_gray,
            (first_gray.shape[1], first_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    first_edges = cv2.Canny(first_gray, 100, 200) > 0
    second_edges = cv2.Canny(second_gray, 100, 200) > 0
    if not first_edges.any() and not second_edges.any():
        return 1.0
    if not first_edges.any() or not second_edges.any():
        return 0.0

    kernel = np.ones((3, 3), dtype=np.uint8)
    first_dilated = cv2.dilate(first_edges.astype(np.uint8), kernel) > 0
    second_dilated = cv2.dilate(second_edges.astype(np.uint8), kernel) > 0
    precision = float(np.mean(first_dilated[second_edges]))
    recall = float(np.mean(second_dilated[first_edges]))
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _color_histogram_distance(first, second, bins=32):
    def histogram(image):
        lab = cv2.cvtColor(_as_uint8_rgb(image), cv2.COLOR_RGB2LAB)
        hist = cv2.calcHist([lab], [1, 2], None, [bins, bins], [0, 256, 0, 256])
        return cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1)

    distance = cv2.compareHist(
        histogram(first), histogram(second), cv2.HISTCMP_BHATTACHARYYA
    )
    return float(np.clip(distance, 0.0, 1.0))


def _patch_descriptors(image, grid_size=8):
    rgb = _as_uint8_rgb(image)
    height = max(grid_size, rgb.shape[0] // grid_size * grid_size)
    width = max(grid_size, rgb.shape[1] // grid_size * grid_size)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

    patch_height = height // grid_size
    patch_width = width // grid_size
    descriptors = []
    for row in range(grid_size):
        for column in range(grid_size):
            row_slice = slice(row * patch_height, (row + 1) * patch_height)
            column_slice = slice(column * patch_width, (column + 1) * patch_width)
            lab_patch = lab[row_slice, column_slice]
            magnitude_patch = magnitude[row_slice, column_slice]
            angle_patch = angle[row_slice, column_slice]
            orientation_hist, _ = np.histogram(
                angle_patch,
                bins=8,
                range=(0.0, 360.0),
                weights=magnitude_patch,
            )
            orientation_hist = orientation_hist.astype(np.float32)
            orientation_hist /= max(float(orientation_hist.sum()), 1e-8)
            descriptor = np.concatenate(
                [
                    lab_patch.mean(axis=(0, 1)),
                    lab_patch.std(axis=(0, 1)),
                    np.array(
                        [magnitude_patch.mean(), magnitude_patch.std()],
                        dtype=np.float32,
                    ),
                    orientation_hist,
                ]
            )
            descriptors.append(descriptor)
    return np.asarray(descriptors, dtype=np.float32)


def _patch_style_metrics(stylized, style):
    stylized_descriptors = torch.from_numpy(_patch_descriptors(stylized))
    style_descriptors = torch.from_numpy(_patch_descriptors(style))
    distances = torch.cdist(stylized_descriptors, style_descriptors)
    nearest_distances, nearest_indices = distances.min(dim=1)
    mean_distance = float(nearest_distances.mean().item())

    counts = torch.bincount(nearest_indices, minlength=style_descriptors.shape[0]).float()
    probabilities = counts[counts > 0] / counts.sum().clamp_min(1.0)
    entropy = -(probabilities * probabilities.log()).sum()
    effective_patch_count = entropy.exp()
    coverage = float((effective_patch_count / style_descriptors.shape[0]).clamp(0.0, 1.0))
    similarity = float(math.exp(-mean_distance))
    return mean_distance, similarity, coverage


def evaluate_single_result(
    content_image,
    style_image,
    stylized_image,
    vae,
    content_features=None,
    style_features=None,
):
    """Return per-pair metrics; all scores are higher-better, distances lower-better."""
    if content_features is None:
        content_features = _extract_vae_features(content_image, vae)
    if style_features is None:
        style_features = _extract_vae_features(style_image, vae)
    stylized_features = _extract_vae_features(stylized_image, vae)

    structure_distance = _self_similarity_distance(
        content_features[-1], stylized_features[-1]
    )
    gram_distance = _multi_scale_gram_distance(style_features, stylized_features)
    color_distance = _color_histogram_distance(style_image, stylized_image)
    patch_distance, patch_similarity, patch_coverage = _patch_style_metrics(
        stylized_image, style_image
    )

    metrics = {
        "content_structure_distance": structure_distance,
        "content_structure_score": 1.0 - structure_distance,
        "content_ssim_gray": _gray_ssim(content_image, stylized_image),
        "content_edge_f1": _edge_f1(content_image, stylized_image),
        "style_gram_distance": gram_distance,
        "style_gram_similarity": 1.0 - gram_distance,
        "style_color_distance": color_distance,
        "style_color_similarity": 1.0 - color_distance,
        "style_patch_distance": patch_distance,
        "style_patch_similarity": patch_similarity,
        "style_patch_coverage": patch_coverage,
    }
    metrics["content_score"] = float(
        np.mean(
            [
                metrics["content_structure_score"],
                metrics["content_ssim_gray"],
                metrics["content_edge_f1"],
            ]
        )
    )
    metrics["style_score"] = float(
        np.mean(
            [
                metrics["style_gram_similarity"],
                metrics["style_color_similarity"],
                metrics["style_patch_similarity"],
            ]
        )
    )
    content_score = metrics["content_score"]
    style_score = metrics["style_score"]
    metrics["balanced_score"] = float(
        2.0 * content_score * style_score / max(content_score + style_score, 1e-8)
    )
    return metrics


def build_run_record(cfg, output_path, metrics):
    return {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metric_version": "styleid_vae_pairwise_v1",
        "content_path": os.path.abspath(cfg.cnt_fn),
        "style_path": os.path.abspath(cfg.sty_fn),
        "output_path": os.path.abspath(output_path),
        "ddim_steps": cfg.ddim_steps,
        "resolution": cfg.resolution,
        "gamma": cfg.gamma,
        "temperature": cfg.T,
        "device": cfg.device,
        "precision": cfg.precision,
        "runtime_device": getattr(cfg, "runtime_device", cfg.device),
        "runtime_precision": getattr(cfg, "runtime_precision", cfg.precision),
        "layers": ",".join(str(layer) for layer in cfg.layers),
        "initial_adain": not cfg.without_init_adain,
        "attention_injection": not cfg.without_attn_injection,
        "trajectory_mode": getattr(cfg, "trajectory_mode", "baseline"),
        "trajectory_drift_threshold": getattr(
            cfg, "trajectory_drift_threshold", 0.4
        ),
        "trajectory_min_generation_weight": getattr(
            cfg, "trajectory_min_generation_weight", 0.1
        ),
        "trajectory_max_generation_weight": getattr(
            cfg, "trajectory_max_generation_weight", 0.4
        ),
        "trajectory_guard_strength": getattr(
            cfg, "trajectory_guard_strength", 0.0
        ),
        "trajectory_min_style_gate": getattr(
            cfg, "trajectory_min_style_gate", 1.0
        ),
        "trajectory_start_fraction": getattr(
            cfg, "trajectory_start_fraction", 0.2
        ),
        "trajectory_end_fraction": getattr(
            cfg, "trajectory_end_fraction", 0.8
        ),
        "trajectory_token_temperature": getattr(
            cfg, "trajectory_token_temperature", 1.0
        ),
        "trajectory_temporal_smoothing": getattr(
            cfg, "trajectory_temporal_smoothing", 0.0
        ),
        "trajectory_output_strength": getattr(
            cfg, "trajectory_output_strength", 1.0
        ),
        "transport_prototype_side": getattr(cfg, "transport_prototype_side", 8),
        "transport_temperature": getattr(cfg, "transport_temperature", 0.07),
        "transport_temporal_weight": getattr(
            cfg, "transport_temporal_weight", 0.25
        ),
        "transport_velocity": getattr(cfg, "transport_velocity", 0.5),
        "transport_coverage_strength": getattr(
            cfg, "transport_coverage_strength", 0.3
        ),
        "transport_sinkhorn_iterations": getattr(
            cfg, "transport_sinkhorn_iterations", 3
        ),
        "transport_max_tokens": getattr(cfg, "transport_max_tokens", 1024),
        "transport_min_tokens": getattr(cfg, "transport_min_tokens", 1),
        "transport_confidence_power": getattr(
            cfg, "transport_confidence_power", 0.0
        ),
        **metrics,
    }


def write_run_record(record, save_dir, output_path, csv_name="evaluation_records.csv"):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.splitext(output_path)[0] + ".metrics.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(record, json_file, ensure_ascii=False, indent=2)

    csv_path = os.path.join(save_dir, csv_name)
    fieldnames = list(record.keys())
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return json_path, csv_path


def format_metrics(metrics):
    return "\n".join(
        [
            "Style transfer evaluation",
            f"  Content structure score: {metrics['content_structure_score']:.4f}  (higher is better)",
            f"  Content gray SSIM:       {metrics['content_ssim_gray']:.4f}  (higher is better)",
            f"  Content edge F1:         {metrics['content_edge_f1']:.4f}  (higher is better)",
            f"  Style Gram similarity:   {metrics['style_gram_similarity']:.4f}  (higher is better)",
            f"  Style color similarity:  {metrics['style_color_similarity']:.4f}  (higher is better)",
            f"  Style patch similarity:  {metrics['style_patch_similarity']:.4f}  (higher is better)",
            f"  Style patch coverage:    {metrics['style_patch_coverage']:.4f}  (higher means less patch collapse)",
            f"  Content score:           {metrics['content_score']:.4f}",
            f"  Style score:             {metrics['style_score']:.4f}",
            f"  Balanced score:          {metrics['balanced_score']:.4f}  (diagnostic, not an absolute quality rating)",
        ]
    )
