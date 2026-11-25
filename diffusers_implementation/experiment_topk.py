"""Pre-transfer Qc/Ks top-k experiment against existing StyleID results.

The stored attention statistic is an association feature, not the attention
actually used by the generation pass. Run from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_styleid_batch import DEFAULT_MODEL_PATH, cpu_attention_features, image_paths, output_name
from run_styleid_diffusers import load_input_image, style_transfer_module
from stable_diffusion import encode_latent, load_stable_diffusion
from utils import normalize


METRICS = ("style_patch_similarity", "content_structure_score")
VERSION = "qc_ks_top_fraction_v1"


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contents", nargs="+", default=["data/cnt/00.png"])
    parser.add_argument("--sty_dir", default="data/sty")
    parser.add_argument("--records_dir", default="results/all_default")
    parser.add_argument("--save_dir", default="results/experiment_topk")
    parser.add_argument("--limit_styles", type=int)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--T", type=float, default=1.5)
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 8, 9, 10, 11])
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sd_version", default="1.5")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def topk_head_stats(query, key, temperature, fraction, chunk_size):
    """Full softmax across keys; chunks only query positions, never keys."""
    if query.ndim != 3 or key.ndim != 3 or query.shape[0] != key.shape[0]:
        raise ValueError("Q and K must have matching [heads, tokens, dimensions]")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q and K head dimensions differ")
    count = key.shape[1]
    k = max(1, math.ceil(fraction * count))
    scale = temperature / math.sqrt(query.shape[-1])
    head_rows = []
    with torch.no_grad():
        for head in range(query.shape[0]):
            key_t = key[head].float().transpose(0, 1).contiguous()
            masses = []
            for start in range(0, query.shape[1], chunk_size):
                logits = torch.matmul(query[head, start:start + chunk_size].float(), key_t) * scale
                # logsumexp uses all style keys. Top-k is selected before
                # exponentiation, which is equivalent to selecting after softmax.
                top = torch.topk(logits, k, dim=-1, sorted=False).values
                mass = torch.exp(torch.logsumexp(top, dim=-1) - torch.logsumexp(logits, dim=-1))
                masses.append(mass.cpu())
            masses = torch.cat(masses).numpy()
            q25, q75 = np.quantile(masses, [0.25, 0.75])
            head_rows.append({"head": head, "query_tokens": int(query.shape[1]),
                              "style_tokens": count, "top_k": k,
                              "topk_mean": float(np.mean(masses)),
                              "topk_iqr": float(q75 - q25)})
    return head_rows


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def spearman(x, y):
    x, y = ranks(x), ranks(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlation_stats(x, y, bootstrap, permutations, seed):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho = spearman(x, y)
    result = {"n": len(x), "spearman_rho": rho, "bootstrap_ci_95": None,
              "permutation_p_two_sided": None, "bootstrap_valid": 0}
    if len(x) < 4 or rho is None:
        return result
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap):
        sample = rng.integers(0, len(x), len(x))
        value = spearman(x[sample], y[sample])
        if value is not None:
            estimates.append(value)
    if estimates:
        result["bootstrap_ci_95"] = [float(a) for a in np.quantile(estimates, [0.025, 0.975])]
    result["bootstrap_valid"] = len(estimates)
    # Rank once for the permutation test; ranks are unchanged by permutation.
    xr, yr = ranks(x), ranks(y)
    xr, yr = xr - xr.mean(), yr - yr.mean()
    denom = math.sqrt(float(xr @ xr) * float(yr @ yr))
    extreme = sum(abs(float(xr @ rng.permutation(yr) / denom)) >= abs(rho) - 1e-12
                  for _ in range(permutations))
    result["permutation_p_two_sided"] = (extreme + 1) / (permutations + 1)
    return result


def validate_record(path, content, style, cfg):
    with open(path, encoding="utf-8") as stream:
        record = json.load(stream)
    expected = {"ddim_steps": cfg.ddim_steps, "resolution": cfg.resolution,
                "gamma": cfg.gamma, "temperature": cfg.T,
                "layers": ",".join(str(i) for i in cfg.layers),
                "initial_adain": True, "attention_injection": True,
                "runtime_precision": cfg.runtime_precision,
                "runtime_device": cfg.runtime_device,
                "metric_version": "styleid_vae_pairwise_v1"}
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"{path}: {field}={record.get(field)!r}, expected {value!r}")
    for field, expected_path in (("content_path", content), ("style_path", style)):
        if os.path.normcase(os.path.abspath(record[field])) != os.path.normcase(os.path.abspath(expected_path)):
            raise ValueError(f"{path}: {field} does not match {expected_path}")
    if not os.path.isfile(record["output_path"]):
        raise FileNotFoundError(record["output_path"])
    return record


def analyse(save_dir, contents, styles, cfg):
    summaries, details, analyses = [], [], []
    for content in contents:
        content_id = Path(content).stem
        content_summaries = []
        for style in styles:
            style_id = Path(style).stem
            path = Path(save_dir) / "pairs" / f"c{content_id}_s{style_id}.json"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as stream:
                pair = json.load(stream)
            rows = pair["details"]
            # Each timestep and layer has the same number of heads in this
            # model, but group explicitly to keep the stated equal weighting.
            layer_values = []
            for layer in cfg.layers:
                step_values = []
                for timestep in sorted({r["timestep"] for r in rows if r["layer"] == layer}):
                    head_values = [r["topk_mean"] for r in rows if r["layer"] == layer and r["timestep"] == timestep]
                    step_values.append(float(np.mean(head_values)))
                if len(step_values) != cfg.ddim_steps:
                    raise ValueError(f"Incomplete timestep data in {path}, layer {layer}")
                layer_values.append(float(np.mean(step_values)))
            summary = {"content": content_id, "style": style_id,
                       "topk_mean": float(np.mean(layer_values)),
                       "topk_iqr_mean": float(np.mean([r["topk_iqr"] for r in rows])),
                       "style_patch_similarity": pair["metrics"]["style_patch_similarity"],
                       "content_structure_score": pair["metrics"]["content_structure_score"],
                       "output_path": pair["output_path"]}
            summaries.append(summary)
            content_summaries.append(summary)
            details.extend({"content": content_id, "style": style_id, **row} for row in rows)
        if not content_summaries:
            continue
        for metric in METRICS:
            x = [r["topk_mean"] for r in content_summaries]
            y = [r[metric] for r in content_summaries]
            stats = correlation_stats(x, y, cfg.bootstrap, cfg.permutations, cfg.seed)
            analyses.append({"content": content_id, "metric": metric, **stats})
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x, y, s=25)
            for row in content_summaries:
                ax.annotate(row["style"], (row["topk_mean"], row[metric]), fontsize=6, xytext=(3, 2), textcoords="offset points")
            rho = stats["spearman_rho"]
            ax.set(title=f"Content {content_id}: Spearman rho = {rho:.3f}" if rho is not None else f"Content {content_id}: rho undefined",
                   xlabel=f"Pre-transfer top {100 * cfg.fraction:g}% attention mass", ylabel=metric)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(Path(save_dir) / f"c{content_id}_{metric}_scatter.png", dpi=160)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        for layer in cfg.layers:
            selected = [r for r in details if r["content"] == content_id and r["layer"] == layer]
            timesteps = sorted({r["timestep"] for r in selected}, reverse=True)
            means = [float(np.mean([r["topk_mean"] for r in selected if r["timestep"] == t])) for t in timesteps]
            ax.plot(timesteps, means, marker=".", label=f"layer {layer}")
        ax.invert_xaxis()
        ax.set(xlabel="Scheduler timestep (denoising order)", ylabel="Top attention mass, mean over styles and heads", title=f"Content {content_id}: exploratory layer/timestep curves")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(save_dir) / f"c{content_id}_layer_curves.png", dpi=160)
        plt.close(fig)
    # Correct the two predefined tests per content, not exploratory curves.
    for content in {row["content"] for row in analyses}:
        tests = [row for row in analyses if row["content"] == content and row["permutation_p_two_sided"] is not None]
        tests.sort(key=lambda row: row["permutation_p_two_sided"])
        running = 1.0
        for index in range(len(tests) - 1, -1, -1):
            running = min(running, tests[index]["permutation_p_two_sided"] * len(tests) / (index + 1))
            tests[index]["bh_q_two_tests"] = running
    for row in analyses:
        row.setdefault("bh_q_two_tests", None)
    write_csv(Path(save_dir) / "pair_summary.csv", summaries)
    write_csv(Path(save_dir) / "head_timestep_details.csv", details)
    write_json(Path(save_dir) / "correlations.json", analyses)
    return analyses


def main():
    cfg = args_parser()
    if not (0 < cfg.fraction <= 1 and cfg.chunk_size > 0 and cfg.ddim_steps > 0 and cfg.resolution > 0 and cfg.resolution % 8 == 0 and cfg.T > 0):
        raise ValueError("Invalid fraction, chunk size, steps, resolution or T")
    if cfg.bootstrap < 1 or cfg.permutations < 1:
        raise ValueError("Bootstrap and permutations must be positive")
    cfg.save_intermediates = False
    cfg.runtime_device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device)
    if cfg.runtime_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = torch.float16 if cfg.precision == "float16" and cfg.runtime_device == "cuda" else torch.float32
    cfg.runtime_precision = str(dtype).replace("torch.", "")
    contents = [os.path.abspath(path) for path in cfg.contents]
    styles = [os.path.abspath(path) for path in image_paths(cfg.sty_dir, cfg.limit_styles)]
    if not contents or not styles or len(set(Path(p).stem for p in contents)) != len(contents) or len(set(Path(p).stem for p in styles)) != len(styles):
        raise ValueError("Image lists must be nonempty with unique file stems")
    for path in contents + styles:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    save_dir = Path(cfg.save_dir)
    (save_dir / "pairs").mkdir(parents=True, exist_ok=True)
    hashes = {path: file_hash(path) for path in contents + styles}
    pending = []
    for content in contents:
        for style in styles:
            output = Path(cfg.records_dir) / output_name(content, style, cfg)
            record_path = output.with_suffix(".metrics.json")
            record = validate_record(record_path, content, style, cfg)
            fingerprint = {"version": VERSION, "content_sha256": hashes[content],
                           "style_sha256": hashes[style], "record_sha256": file_hash(record_path),
                           "model_id": os.path.abspath(cfg.model_id), "sd_version": cfg.sd_version,
                           "steps": cfg.ddim_steps, "resolution": cfg.resolution,
                           "layers": cfg.layers, "temperature": cfg.T, "fraction": cfg.fraction,
                           "runtime_device": cfg.runtime_device, "runtime_precision": cfg.runtime_precision}
            pair_path = save_dir / "pairs" / f"c{Path(content).stem}_s{Path(style).stem}.json"
            if pair_path.exists():
                with open(pair_path, encoding="utf-8") as stream:
                    previous = json.load(stream)
                if previous.get("fingerprint") != fingerprint:
                    raise ValueError(f"Existing result differs from current inputs/configuration: {pair_path}; choose a new save_dir")
                continue
            pending.append((content, style, record, fingerprint, pair_path))
    write_json(save_dir / "manifest.json", {"version": VERSION, "parameters": vars(cfg),
               "contents": contents, "styles": styles, "input_sha256": hashes,
               "note": "Existing evaluation records do not include model revision or image hashes; exact historical provenance cannot be verified."})
    print(f"Pairs: {len(contents)} x {len(styles)}; pending: {len(pending)}", flush=True)
    if pending:
        vae, tokenizer, text_encoder, unet, scheduler = load_stable_diffusion(
            sd_version=cfg.sd_version, precision_t=dtype, device=cfg.runtime_device,
            model_id=cfg.model_id, local_files_only=True)
        scheduler.set_timesteps(cfg.ddim_steps)
        wrapper = style_transfer_module(unet, vae, text_encoder, tokenizer, scheduler, cfg,
                                        {"gamma": cfg.gamma, "tau": cfg.T, "injection_layers": cfg.layers})
        denoise_kwargs = wrapper.get_text_condition(None)
        text_encoder.to("cpu")
        content_features = {}
        for content in {item[0] for item in pending}:
            print(f"Inverting content {Path(content).name}", flush=True)
            image = load_input_image(content, cfg.resolution)
            latent = encode_latent(normalize(image).to(device=vae.device, dtype=dtype), vae)
            wrapper.trigger_get_qkv, wrapper.trigger_modify_qkv, wrapper.capture_mode = True, False, "content"
            _, latents = wrapper.invert_process(latent, denoise_kwargs)
            content_features[content] = cpu_attention_features(wrapper.take_attn_features(), "content")
            del latent, latents
        started = time.time()
        for style in styles:
            batch = [item for item in pending if item[1] == style]
            if not batch:
                continue
            print(f"Inverting style {Path(style).name}", flush=True)
            image = load_input_image(style, cfg.resolution)
            latent = encode_latent(normalize(image).to(device=vae.device, dtype=dtype), vae)
            wrapper.trigger_get_qkv, wrapper.trigger_modify_qkv, wrapper.capture_mode = True, False, "style"
            _, latents = wrapper.invert_process(latent, denoise_kwargs)
            style_features = wrapper.take_attn_features()
            del latent, latents
            for content, _, record, fingerprint, pair_path in batch:
                details = []
                for layer in cfg.layers:
                    name = f"layer{layer}_attn"
                    for timestep in sorted(style_features[name], reverse=True):
                        q = content_features[content][name][timestep][0].to(cfg.runtime_device)
                        key = style_features[name][timestep][1]
                        for row in topk_head_stats(q, key, cfg.T, cfg.fraction, cfg.chunk_size):
                            details.append({"layer": layer, "timestep": int(timestep), **row})
                        del q, key
                pair = {"fingerprint": fingerprint, "metrics": {m: float(record[m]) for m in METRICS},
                        "output_path": record["output_path"], "details": details}
                write_json(pair_path, pair)
                print(f"Saved {pair_path.name}; elapsed {(time.time() - started)/60:.1f} min", flush=True)
            del style_features
            if cfg.runtime_device == "cuda":
                torch.cuda.empty_cache()
    analyses = analyse(save_dir, contents, styles, cfg)
    for row in analyses:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"Results: {save_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
