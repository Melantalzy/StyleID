"""Post-hoc follow-up: input color similarity vs edge-F1 injection effect."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_artflow_new_styles import read, records
from experiment_topk import correlation_stats, spearman, write_csv, write_json


def main():
    on = records("results/all_default/evaluation_records.csv", True)
    off = records("results/attention_disabled_c00/evaluation_records.csv", False)
    if set(on) != set(off) or len(on) != 920:
        raise ValueError("Expected complete paired dataset")
    features = {(r["content"], r["style"]): float(r["input_color_similarity"])
                for r in read("results/experiment_baseline_gain/paired_baseline_gain.csv")}
    pairs = []
    for key in sorted(on):
        a, b = on[key], off[key]
        c, s = Path(a["content_path"]).stem, Path(a["style_path"]).stem
        pairs.append({"content": c, "style": s, "input_color_similarity": features[(c, s)],
                      "edge_f1_delta": float(a["content_edge_f1"]) - float(b["content_edge_f1"])})
    contents = sorted({r["content"] for r in pairs})
    styles = sorted({r["style"] for r in pairs})
    lookup = {(r["content"], r["style"]): r for r in pairs}
    x = np.array([[lookup[(c, s)]["input_color_similarity"] for s in styles] for c in contents])
    y = np.array([[lookup[(c, s)]["edge_f1_delta"] for s in styles] for c in contents])
    group = []
    for c, a, b in zip(contents, x, y):
        result = correlation_stats(a, b, 3000, 5000, 719)
        ci = result["bootstrap_ci_95"]
        group.append({"content": c, "rho": result["spearman_rho"], "ci_low": ci[0] if ci else None,
                      "ci_high": ci[1] if ci else None, "p_unadjusted": result["permutation_p_two_sided"]})
    rng = np.random.default_rng(719)
    boot = []
    for _ in range(3000):
        ii = rng.integers(0, len(contents), len(contents))
        jj = rng.integers(0, len(styles), len(styles))
        rhos = [spearman(x[i, jj], y[i, jj]) for i in ii]
        rhos = [r for r in rhos if r is not None]
        if rhos:
            boot.append(float(np.mean(rhos)))
    rho = [r["rho"] for r in group if r["rho"] is not None]
    summary = {"mean_rho": float(np.mean(rho)), "positive": sum(v > 0 for v in rho),
               "negative": sum(v < 0 for v in rho), "n_content": len(contents), "n_style": len(styles),
               "crossed_bootstrap_ci": [float(v) for v in np.quantile(boot, [.025, .975])],
               "limitations": "Post-hoc hypothesis selected on a different, previously analysed dataset; this dataset also analysed in many prior experiments. Same 46 styles across content, not independent replications. F1 is unvalidated and style can change brightness/edges. Not evidence of causality."}
    out = Path("results/experiment_color_edge_followup")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pairs.csv", pairs)
    write_csv(out / "per_content.csv", group)
    write_json(out / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
