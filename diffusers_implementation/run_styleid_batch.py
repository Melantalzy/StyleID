"""Efficient, resumable StyleID batch generation for content/style folders."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from run_styleid_diffusers import (
    load_input_image,
    style_transfer_module,
    write_trajectory_diagnostics,
)
from single_image_metrics import (
    _extract_vae_features,
    build_run_record,
    evaluate_single_result,
    format_metrics,
    write_run_record,
)
from stable_diffusion import encode_latent, load_stable_diffusion
from utils import denormalize, normalize, save_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_MODEL_PATH = os.path.join(
    ".cache",
    "huggingface",
    "hub",
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5",
    "snapshots",
    "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
)


def get_args():
    parser = argparse.ArgumentParser(description="Run StyleID for every content/style pair")
    parser.add_argument("--cnt_dir", default="data/cnt")
    parser.add_argument("--sty_dir", default="data/sty")
    parser.add_argument("--save_dir", default="results/all_default")
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--T", type=float, default=1.5)
    parser.add_argument("--layers", nargs="+", type=int, default=[7, 8, 9, 10, 11])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--sd_version", default="1.5")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--without_init_adain", action="store_true")
    parser.add_argument("--without_attn_injection", action="store_true")
    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument(
        "--trajectory_mode",
        choices=[
            "baseline", "dynamic", "token_dynamic", "output_fusion", "transport"
        ],
        default="baseline",
    )
    parser.add_argument("--trajectory_drift_threshold", type=float, default=0.4)
    parser.add_argument(
        "--trajectory_min_generation_weight", type=float, default=0.1
    )
    parser.add_argument(
        "--trajectory_max_generation_weight", type=float, default=0.4
    )
    parser.add_argument("--trajectory_guard_strength", type=float, default=0.0)
    parser.add_argument("--trajectory_min_style_gate", type=float, default=1.0)
    parser.add_argument("--trajectory_start_fraction", type=float, default=0.2)
    parser.add_argument("--trajectory_end_fraction", type=float, default=0.8)
    parser.add_argument("--trajectory_token_temperature", type=float, default=1.0)
    parser.add_argument("--trajectory_temporal_smoothing", type=float, default=0.0)
    parser.add_argument("--trajectory_output_strength", type=float, default=1.0)
    parser.add_argument("--transport_prototype_side", type=int, default=8)
    parser.add_argument("--transport_temperature", type=float, default=0.07)
    parser.add_argument("--transport_temporal_weight", type=float, default=0.25)
    parser.add_argument("--transport_velocity", type=float, default=0.5)
    parser.add_argument("--transport_coverage_strength", type=float, default=0.3)
    parser.add_argument("--transport_sinkhorn_iterations", type=int, default=3)
    parser.add_argument("--transport_max_tokens", type=int, default=1024)
    parser.add_argument("--transport_min_tokens", type=int, default=1)
    parser.add_argument("--transport_confidence_power", type=float, default=0.0)
    parser.add_argument("--save_trajectory_diagnostics", action="store_true")
    parser.add_argument("--limit_contents", type=int, default=None)
    parser.add_argument("--limit_styles", type=int, default=None)
    return parser.parse_args()


def image_paths(directory, limit=None):
    paths = [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    return paths if limit is None else paths[:limit]


def cpu_attention_features(features, mode):
    copied = {}
    for layer_name, timestep_features in features.items():
        copied[layer_name] = {}
        for timestep, (query, key, value) in timestep_features.items():
            if mode == "content":
                copied[layer_name][timestep] = (
                    query.detach().to("cpu"),
                    None,
                    None,
                )
            else:
                copied[layer_name][timestep] = (
                    None,
                    key.detach().to("cpu"),
                    value.detach().to("cpu"),
                )
    return copied


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value if item is not None)
    return 0


def tag_number(value):
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def output_name(content_path, style_path, cfg):
    content_stem = os.path.splitext(os.path.basename(content_path))[0]
    style_stem = os.path.splitext(os.path.basename(style_path))[0]
    return (
        f"c{content_stem}_s{style_stem}_d{cfg.ddim_steps}_r{cfg.resolution}_"
        f"g{tag_number(cfg.gamma)}_t{tag_number(cfg.T)}.jpg"
    )


def move_content_features_to_device(content_features, style_features, device):
    result = {}
    for layer_name, timestep_features in style_features.items():
        result[layer_name] = {}
        for timestep, (_, style_key, style_value) in timestep_features.items():
            content_query = content_features[layer_name][timestep][0].to(
                device=device, non_blocking=True
            )
            result[layer_name][timestep] = (
                content_query,
                style_key,
                style_value,
            )
    return result


def main():
    cfg = get_args()
    if cfg.resolution <= 0 or cfg.resolution % 8 != 0:
        raise ValueError("--resolution must be a positive multiple of 8")
    if not 0.0 <= cfg.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1")
    if cfg.T <= 0:
        raise ValueError("--T must be greater than 0")
    if cfg.trajectory_drift_threshold <= 0:
        raise ValueError("--trajectory_drift_threshold must be greater than 0")
    base_generation_weight = 1.0 - cfg.gamma
    if (
        cfg.trajectory_mode in (
            "dynamic", "token_dynamic", "output_fusion", "transport"
        )
        and not 0.0
        <= cfg.trajectory_min_generation_weight
        <= base_generation_weight
        <= cfg.trajectory_max_generation_weight
        <= 1.0
    ):
        raise ValueError(
            "trajectory generation weights must satisfy 0 <= min <= "
            f"{base_generation_weight:.4f} <= max <= 1"
        )
    if cfg.trajectory_guard_strength < 0:
        raise ValueError("--trajectory_guard_strength must be non-negative")
    if not 0.0 <= cfg.trajectory_min_style_gate <= 1.0:
        raise ValueError("--trajectory_min_style_gate must be between 0 and 1")
    if not (
        0.0
        <= cfg.trajectory_start_fraction
        <= cfg.trajectory_end_fraction
        <= 1.0
    ):
        raise ValueError(
            "trajectory fractions must satisfy 0 <= start <= end <= 1"
        )
    if cfg.trajectory_token_temperature <= 0:
        raise ValueError("--trajectory_token_temperature must be greater than 0")
    if not 0.0 <= cfg.trajectory_temporal_smoothing < 1.0:
        raise ValueError(
            "--trajectory_temporal_smoothing must satisfy 0 <= value < 1"
        )
    if cfg.trajectory_output_strength < 0.0:
        raise ValueError("--trajectory_output_strength must be non-negative")
    if cfg.transport_prototype_side <= 0:
        raise ValueError("--transport_prototype_side must be greater than 0")
    if cfg.transport_temperature <= 0.0:
        raise ValueError("--transport_temperature must be greater than 0")
    if not 0.0 <= cfg.transport_temporal_weight <= 1.0:
        raise ValueError("--transport_temporal_weight must be between 0 and 1")
    if cfg.transport_velocity < 0.0:
        raise ValueError("--transport_velocity must be non-negative")
    if not 0.0 <= cfg.transport_coverage_strength <= 1.0:
        raise ValueError("--transport_coverage_strength must be between 0 and 1")
    if cfg.transport_sinkhorn_iterations < 0:
        raise ValueError("--transport_sinkhorn_iterations must be non-negative")
    if cfg.transport_max_tokens <= 0:
        raise ValueError("--transport_max_tokens must be greater than 0")
    if not 0 < cfg.transport_min_tokens <= cfg.transport_max_tokens:
        raise ValueError(
            "transport token limits must satisfy 0 < min_tokens <= max_tokens"
        )
    if cfg.transport_confidence_power < 0.0:
        raise ValueError("--transport_confidence_power must be non-negative")

    content_paths = image_paths(cfg.cnt_dir, cfg.limit_contents)
    style_paths = image_paths(cfg.sty_dir, cfg.limit_styles)
    if not content_paths or not style_paths:
        raise ValueError("The content and style directories must both contain images")

    os.makedirs(cfg.save_dir, exist_ok=True)
    total_pairs = len(content_paths) * len(style_paths)
    print(
        f"Batch: {len(content_paths)} content images x {len(style_paths)} style images "
        f"= {total_pairs} pairs"
    )

    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = torch.float16 if cfg.precision == "float16" else torch.float32
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    cfg.runtime_device = device
    cfg.runtime_precision = str(dtype).replace("torch.", "")

    vae, tokenizer, text_encoder, unet, scheduler = load_stable_diffusion(
        sd_version=cfg.sd_version,
        precision_t=dtype,
        device=device,
        model_id=cfg.model_id,
        local_files_only=cfg.local_files_only,
    )
    scheduler.set_timesteps(cfg.ddim_steps)
    style_transfer_params = {
        "gamma": cfg.gamma,
        "tau": cfg.T,
        "injection_layers": cfg.layers,
        "trajectory_mode": cfg.trajectory_mode,
        "trajectory_drift_threshold": cfg.trajectory_drift_threshold,
        "trajectory_min_generation_weight": cfg.trajectory_min_generation_weight,
        "trajectory_max_generation_weight": cfg.trajectory_max_generation_weight,
        "trajectory_guard_strength": cfg.trajectory_guard_strength,
        "trajectory_min_style_gate": cfg.trajectory_min_style_gate,
        "trajectory_start_fraction": cfg.trajectory_start_fraction,
        "trajectory_end_fraction": cfg.trajectory_end_fraction,
        "trajectory_token_temperature": cfg.trajectory_token_temperature,
        "trajectory_temporal_smoothing": cfg.trajectory_temporal_smoothing,
        "trajectory_output_strength": cfg.trajectory_output_strength,
        "transport_prototype_side": cfg.transport_prototype_side,
        "transport_temperature": cfg.transport_temperature,
        "transport_temporal_weight": cfg.transport_temporal_weight,
        "transport_velocity": cfg.transport_velocity,
        "transport_coverage_strength": cfg.transport_coverage_strength,
        "transport_sinkhorn_iterations": cfg.transport_sinkhorn_iterations,
        "transport_max_tokens": cfg.transport_max_tokens,
        "transport_min_tokens": cfg.transport_min_tokens,
        "transport_confidence_power": cfg.transport_confidence_power,
        "save_trajectory_diagnostics": cfg.save_trajectory_diagnostics,
    }
    wrapper = style_transfer_module(
        unet,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        cfg,
        style_transfer_params=style_transfer_params,
    )
    denoise_kwargs = wrapper.get_text_condition(None)
    text_encoder.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Precomputing content inversions...")
    contents = []
    for index, path in enumerate(content_paths, start=1):
        image = load_input_image(path, cfg.resolution)
        latent = encode_latent(normalize(image).to(device=vae.device, dtype=dtype), vae)
        wrapper.trigger_get_qkv = True
        wrapper.trigger_modify_qkv = False
        wrapper.capture_mode = "content"
        _, latents = wrapper.invert_process(latent, denoise_kwargs=denoise_kwargs)
        attention = cpu_attention_features(wrapper.take_attn_features(), "content")
        reference_features = _extract_vae_features(image, vae)
        contents.append(
            {
                "path": path,
                "image": image,
                "latent": latents[-1].detach().to("cpu"),
                "attention": attention,
                "reference_features": reference_features,
            }
        )
        del latent, latents
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  Content {index}/{len(content_paths)}: {os.path.basename(path)}")

    content_gib = sum(tensor_bytes(item["attention"]) for item in contents) / 1024**3
    print(f"Cached content attention features in RAM: {content_gib:.2f} GiB")

    completed = 0
    skipped = 0
    started_at = time.time()
    for style_index, style_path in enumerate(style_paths, start=1):
        pending_contents = []
        for content in contents:
            name = output_name(content["path"], style_path, cfg)
            output_path = os.path.join(cfg.save_dir, name)
            json_path = os.path.splitext(output_path)[0] + ".metrics.json"
            if os.path.exists(output_path) and os.path.exists(json_path):
                skipped += 1
            else:
                pending_contents.append((content, output_path))
        if not pending_contents:
            print(
                f"Style {style_index}/{len(style_paths)} already complete: "
                f"{os.path.basename(style_path)}"
            )
            continue

        print(
            f"Inverting style {style_index}/{len(style_paths)}: "
            f"{os.path.basename(style_path)}"
        )
        style_image = load_input_image(style_path, cfg.resolution)
        style_latent_input = encode_latent(
            normalize(style_image).to(device=vae.device, dtype=dtype), vae
        )
        wrapper.trigger_get_qkv = True
        wrapper.trigger_modify_qkv = False
        wrapper.capture_mode = "style"
        _, style_latents = wrapper.invert_process(
            style_latent_input, denoise_kwargs=denoise_kwargs
        )
        style_latent = style_latents[-1]
        style_attention = wrapper.take_attn_features()
        style_reference_features = _extract_vae_features(style_image, vae)
        del style_latent_input, style_latents

        for content, output_path in pending_contents:
            wrapper.attn_features_modify = move_content_features_to_device(
                content["attention"], style_attention, device
            )
            wrapper.trigger_get_qkv = False
            wrapper.trigger_modify_qkv = not cfg.without_attn_injection
            wrapper.capture_mode = None

            content_latent = content["latent"].to(device=device, dtype=dtype)
            if cfg.without_init_adain:
                combined_latent = content_latent
            else:
                combined_latent = (
                    (content_latent - content_latent.mean(dim=(2, 3), keepdim=True))
                    / (content_latent.std(dim=(2, 3), keepdim=True) + 1e-4)
                    * style_latent.std(dim=(2, 3), keepdim=True)
                    + style_latent.mean(dim=(2, 3), keepdim=True)
                )

            wrapper.reset_trajectory_diagnostics()
            images, generated_latents = wrapper.reverse_process(
                combined_latent, denoise_kwargs=denoise_kwargs
            )
            trajectory_diagnostics = wrapper.take_trajectory_diagnostics()
            stylized_image = denormalize(images[-1])[0]
            save_image(stylized_image, output_path)
            if cfg.save_trajectory_diagnostics:
                write_trajectory_diagnostics(
                    trajectory_diagnostics, cfg, output_path
                )

            metrics = evaluate_single_result(
                content_image=content["image"],
                style_image=style_image,
                stylized_image=stylized_image,
                vae=vae,
                content_features=content["reference_features"],
                style_features=style_reference_features,
            )
            cfg.cnt_fn = content["path"]
            cfg.sty_fn = style_path
            record = build_run_record(cfg, output_path, metrics)
            json_path, csv_path = write_run_record(record, cfg.save_dir, output_path)

            completed += 1
            processed = completed + skipped
            elapsed = time.time() - started_at
            average = elapsed / max(completed, 1)
            remaining = max(total_pairs - processed, 0)
            eta_minutes = average * remaining / 60.0
            print(
                f"[{processed}/{total_pairs}] {os.path.basename(output_path)} | "
                f"content={metrics['content_score']:.4f} "
                f"style={metrics['style_score']:.4f} "
                f"balanced={metrics['balanced_score']:.4f} | "
                f"ETA {eta_minutes:.1f} min"
            )

            del (
                content_latent,
                combined_latent,
                images,
                generated_latents,
                stylized_image,
            )
            wrapper.attn_features_modify = {
                layer_name: {} for layer_name in style_attention
            }
            if device == "cuda":
                torch.cuda.empty_cache()

        del style_latent, style_attention, style_reference_features
        if device == "cuda":
            torch.cuda.empty_cache()

    elapsed_minutes = (time.time() - started_at) / 60.0
    print(
        f"Batch complete: generated {completed}, skipped {skipped}, "
        f"total {total_pairs}, elapsed {elapsed_minutes:.1f} min"
    )
    print(f"Results: {os.path.abspath(cfg.save_dir)}")
    print(f"Evaluation CSV: {os.path.abspath(os.path.join(cfg.save_dir, 'evaluation_records.csv'))}")


if __name__ == "__main__":
    main()
