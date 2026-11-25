"""Supplementary CSD style similarity on frozen new-style follow-up pairs.

Requires the inspected local MIT CSD port and CC-BY-4.0 weights downloaded to
downloads/csd-port. Upstream authors warn of checkpoint/paper discrepancies.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPImageProcessor

from experiment_artflow_new_styles import read
from experiment_topk import spearman, write_csv, write_json
from run_styleid_diffusers import load_input_image


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def main():
    model_dir = Path("downloads/csd-port")
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this CSD experiment")
    sys.path.insert(0, str(model_dir.resolve()))
    from modeling_csd import CSDConfig, CSDModel

    model = CSDModel(CSDConfig())
    state = load_file(str(weights), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model = model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float16)
    processor = CLIPImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    pairs = read("results/experiment_artflow_new_styles/paired_conditions.csv")
    if len(pairs) != 280:
        raise ValueError("Expected frozen 10x28 follow-up pairs")
    styles = {r["style"]: Path("downloads/style-transfer-dataset/styles") / f"style_{r['style']}.jpg" for r in pairs}
    jobs = [(f"style:{sid}", path, True) for sid, path in styles.items()]
    jobs += [(f"on:{r['content']}:{r['style']}", Path(r["on_output"]), False) for r in pairs]
    jobs += [(f"off:{r['content']}:{r['style']}", Path(r["off_output"]), False) for r in pairs]
    if any(not path.is_file() for _, path, _ in jobs):
        raise FileNotFoundError("A style or output image is missing")
    features = {}
    batch_size = 4
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start+batch_size]
        images = []
        for _, path, crop in batch:
            if crop:
                images.append(Image.fromarray(load_input_image(str(path), 512)))
            else:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            vectors = model(pixel_values=inputs).embeddings.float().cpu().numpy()
        for (key, _, _), vector in zip(batch, vectors):
            features[key] = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if (start // batch_size) % 15 == 0:
            print(f"CSD embeddings {min(start + batch_size, len(jobs))}/{len(jobs)}", flush=True)
    rows = []
    for r in pairs:
        cid, sid = r["content"], r["style"]
        style = features[f"style:{sid}"]
        on = float(np.dot(features[f"on:{cid}:{sid}"], style))
        off = float(np.dot(features[f"off:{cid}:{sid}"], style))
        rows.append({"content": cid, "style": sid, "input_patch_similarity": float(r["input_patch_similarity"]),
                     "csd_on": on, "csd_off": off, "csd_delta": on-off})
    contents = sorted({r["content"] for r in rows}, key=int)
    style_ids = sorted({r["style"] for r in rows}, key=int)
    lookup = {(r["content"], r["style"]): r for r in rows}
    x = np.array([[lookup[(c, s)]["input_patch_similarity"] for s in style_ids] for c in contents])
    rng = np.random.default_rng(145)
    summary = []
    for target in ("csd_on", "csd_delta"):
        y = np.array([[lookup[(c, s)][target] for s in style_ids] for c in contents])
        group_rho = [spearman(a, b) for a, b in zip(x, y)]
        boot = []
        for _ in range(3000):
            ci = rng.integers(0, len(contents), len(contents))
            si = rng.integers(0, len(style_ids), len(style_ids))
            values = [spearman(x[i, si], y[i, si]) for i in ci]
            values = [r for r in values if r is not None]
            if values:
                boot.append(float(np.mean(values)))
        low, high = np.quantile(boot, [.025, .975])
        summary.append({"target": target, "mean_rho": float(np.mean(group_rho)),
                        "positive": sum(r > 0 for r in group_rho), "negative": sum(r < 0 for r in group_rho),
                        "contents": len(contents), "styles": len(style_ids),
                        "ci_low": float(low), "ci_high": float(high)})
        print(summary[-1], flush=True)
    out = Path("results/experiment_csd_style")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pair_scores.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_json(out / "manifest.json", {"source": "bigshanedogg/CSD c3ffe95 unofficial port; weights from tomg-group-umd/CSD-ViT-L",
        "weights_sha256": sha256(weights), "device": "cuda", "dtype": "float16",
        "scope": "frozen 10 contents x 28 new styles; CSD cosine similarity of output and cropped reference",
        "limitations": "Upstream author warns public weights may not reproduce reported results. CSD is an automated style proxy, not human rating. JPEG outputs; post-hoc exploratory; no multiplicity correction, 10 contents and 28 shared styles, not causal evidence."})
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
