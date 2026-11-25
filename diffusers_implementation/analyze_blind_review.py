"""Analyse exported local blinded-review JSON files after human scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment_topk import correlation_stats, spearman, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratings", nargs="+", help="JSON files exported by blind review page")
    parser.add_argument("--save_dir", default="results/blind_review_analysis")
    cfg = parser.parse_args()
    key = json.loads(Path("results/blind_review_pilot/answer_key.json").read_text(encoding="utf-8"))
    if len(key) != 12 or len({r["id"] for r in key}) != 12:
        raise ValueError("Expected 12 unique blind-review keys")
    valid_scores = set(range(1, 6))
    rows, raters = [], set()
    for source in cfg.ratings:
        rating = json.loads(Path(source).read_text(encoding="utf-8"))
        rater = rating.get("rater", "").strip()
        if rating.get("version") != "styleid_blind_review_v1" or not rater or rater in raters:
            raise ValueError(f"Wrong version or duplicate/empty rater ID: {source}")
        raters.add(rater)
        scores = rating["scores"]
        expected = {f"{case['id']}_{arm}_{dim}" for case in key for arm in ("A", "B") for dim in ("content", "style", "artifact")}
        if set(scores) != expected or any(type(value) is not int or value not in valid_scores for value in scores.values()):
            raise ValueError(f"Incomplete or invalid ratings in {source}")
        for case in key:
            for arm in ("A", "B"):
                rows.append({"rater": rater, "case": case["id"], "dataset": case["dataset"],
                             "content_image": case["content"], "style_image": case["style"],
                             "input_patch_similarity": float(case["input_patch_similarity"]),
                             "condition": case[arm],
                             **{dim: scores[f"{case['id']}_{arm}_{dim}"] for dim in ("content", "style", "artifact")}})
    summary, paired_rows = [], []
    for dim in ("content", "style", "artifact"):
        for condition in ("on", "off"):
            per_case = []
            for case in key:
                sample = [r[dim] for r in rows if r["case"] == case["id"] and r["condition"] == condition]
                per_case.append({"case": case["id"], "feature": float(case["input_patch_similarity"]),
                                 "mean_rating": float(np.mean(sample))})
            rho = spearman([v["feature"] for v in per_case], [v["mean_rating"] for v in per_case])
            summary.append({"dimension": dim, "condition": condition, "n_cases": len(per_case),
                            "n_raters": len(raters), "spearman_rho": rho,
                            "mean_rating": float(np.mean([v["mean_rating"] for v in per_case]))})
        for case in key:
            on = np.mean([r[dim] for r in rows if r["case"] == case["id"] and r["condition"] == "on"])
            off = np.mean([r[dim] for r in rows if r["case"] == case["id"] and r["condition"] == "off"])
            paired_rows.append({"dimension": dim, "case": case["id"],
                                "input_patch_similarity": float(case["input_patch_similarity"]),
                                "on": float(on), "off": float(off), "delta_on_minus_off": float(on - off)})
    out = Path(cfg.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "unblinded_ratings.csv").exists():
        raise FileExistsError(f"Use a new --save_dir to avoid overwriting existing analysis: {out}")
    write_csv(out / "unblinded_ratings.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_ratings.csv", paired_rows)
    paired_summary = []
    for dim in ("content", "style", "artifact"):
        group = [r for r in paired_rows if r["dimension"] == dim]
        delta = np.array([r["delta_on_minus_off"] for r in group])
        result = correlation_stats([r["input_patch_similarity"] for r in group], delta, 5000, 10000, 42)
        interval = result["bootstrap_ci_95"]
        paired_summary.append({"dimension": dim, "n_cases": len(group),
            "mean_delta": float(np.mean(delta)), "median_delta": float(np.median(delta)),
            "on_higher": int(np.sum(delta > 0)), "off_higher": int(np.sum(delta < 0)),
            "tie": int(np.sum(delta == 0)), "similarity_vs_delta_rho": result["spearman_rho"],
            "rho_ci_low": interval[0] if interval else None, "rho_ci_high": interval[1] if interval else None,
            "p_unadjusted": result["permutation_p_two_sided"]})
    write_csv(out / "paired_summary.csv", paired_summary)
    write_json(out / "manifest.json", {"case_count": 12, "rater_count": len(raters),
               "limitations": "12-case pilot only, no external test or inferential claim. Diverse images are not a representative random sample; scores are ordinal and rater preferences may vary."})
    print(f"Analysed {len(raters)} raters, 12 paired cases: {out.resolve()}")


if __name__ == "__main__":
    main()
