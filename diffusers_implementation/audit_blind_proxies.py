"""Exploratory proxy-vs-human agreement on the 12 scored StyleID outputs."""
from __future__ import annotations

import csv
from pathlib import Path

from experiment_topk import correlation_stats, write_csv, write_json


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    human = [r for r in read("results/blind_review_analysis_paired/unblinded_ratings.csv") if r["condition"] == "on"]
    def idx(path):
        return {(r["dataset"], r["content"], r["style"]): r for r in read(path)}
    lpips = idx("results/experiment_lpips_content/pair_scores.csv")
    gram = idx("results/experiment_alexnet_gram/pair_scores.csv")
    eval_rows = []
    for path in ("results/all_default/evaluation_records.csv", "results/external_eval_on/evaluation_records.csv",
                 "results/artflow_inputs_styleid_on/evaluation_records.csv"):
        eval_rows.extend(read(path))
    eval_by_path = {str(Path(r["output_path"]).resolve()).casefold(): r for r in eval_rows}
    combined = []
    for h in human:
        key = (h["dataset"], h["content_image"], h["style_image"])
        l, g = lpips[key], gram[key]
        record = eval_by_path[str(Path(l["on_output"]).resolve()).casefold()]
        combined.append({"case": h["case"], "content_rating": int(h["content"]),
                         "style_rating": int(h["style"]), "artifact_rating": int(h["artifact"]),
                         "input_patch_similarity": float(h["input_patch_similarity"]),
                         "content_structure_score": float(record["content_structure_score"]),
                         "content_ssim_gray": float(record["content_ssim_gray"]),
                         "content_edge_f1": float(record["content_edge_f1"]),
                         "lpips_content_similarity": -float(l["lpips_on"]),
                         "style_patch_similarity": float(record["style_patch_similarity"]),
                         "style_gram_similarity": float(record["style_gram_similarity"]),
                         "style_color_similarity": float(record["style_color_similarity"]),
                         "alex_gram_similarity": float(g["alex_gram_on"])})
    hypotheses = [("content_rating", f) for f in ("input_patch_similarity", "content_structure_score", "content_ssim_gray", "content_edge_f1", "lpips_content_similarity")]
    hypotheses += [("style_rating", f) for f in ("input_patch_similarity", "style_patch_similarity", "style_gram_similarity", "style_color_similarity", "alex_gram_similarity")]
    summary = []
    for rating, feature in hypotheses:
        stats = correlation_stats([r[feature] for r in combined], [r[rating] for r in combined], 5000, 10000, 123)
        ci = stats["bootstrap_ci_95"]
        summary.append({"rating": rating, "feature": feature, "n": stats["n"], "rho": stats["spearman_rho"],
                        "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
                        "p_unadjusted": stats["permutation_p_two_sided"]})
    out = Path("results/blind_review_proxy_audit")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "combined.csv", combined)
    write_csv(out / "correlations.csv", summary)
    write_json(out / "manifest.json", {"status": "exploratory 12-case pilot, one rater, 10 comparisons; no multiplicity correction",
               "limits": "A single rater and tiny sample with tied ordinal scores; broad intervals and possible selection bias. Proxy correlation is not validation or evidence of causation."})
    for row in summary:
        print(row)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
