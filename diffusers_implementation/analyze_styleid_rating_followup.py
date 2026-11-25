"""Analyze the local StyleID rating pilot within each content image."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def spearman(x, y):
    def ranks(values):
        values = np.asarray(values)
        return np.array([np.mean(np.flatnonzero(np.sort(values) == v)) for v in values])

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--key", type=Path, default=Path("results/styleid_rating_followup/answer_key.json"))
    parser.add_argument("--save_dir", type=Path, default=Path("results/styleid_rating_followup_analysis"))
    args = parser.parse_args()
    cases = json.loads(args.key.read_text(encoding="utf-8"))
    dimensions = ("content", "style")
    expected = {f"{case['id']}_{dim}" for case in cases for dim in ("content", "style", "artifact")}
    raters = set()
    ratings = defaultdict(list)
    for path in args.scores:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("version") != "styleid_rating_followup_v1":
            raise ValueError(f"Unexpected score format: {path}")
        rater = data.get("rater", "").strip()
        if not rater or rater in raters:
            raise ValueError(f"Missing or repeated rater ID: {rater!r}")
        raters.add(rater)
        scores = data.get("scores", {})
        if set(scores) != expected:
            raise ValueError(f"Missing or unexpected score keys in {path}")
        for key, value in scores.items():
            if type(value) is not int or not 1 <= value <= 5:
                raise ValueError(f"Invalid score {key}: {value!r}")
            ratings[key].append(value)
    rows = []
    for case in cases:
        row = dict(case)
        for dim in dimensions:
            row[f"rating_{dim}"] = float(np.mean(ratings[f"{case['id']}_{dim}"]))
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['content'])].append(row)
    per_content = []
    for (dataset, content), group in sorted(groups.items()):
        for dim in dimensions:
            x = [row['input_patch_similarity'] for row in group]
            y = [row[f"rating_{dim}"] for row in group]
            rho = spearman(x, y) if len(set(x)) > 1 and len(set(y)) > 1 else float('nan')
            per_content.append(dict(dataset=dataset, content=content, dimension=dim, n=len(group), rho=rho))
    summaries = []
    for dim in dimensions:
        values = [row['rho'] for row in per_content if row['dimension'] == dim and np.isfinite(row['rho'])]
        summaries.append(dict(dimension=dim, valid_contents=len(values), total_contents=len(groups),
                              mean_rho=float(np.mean(values)) if values else float('nan'),
                              positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values),
                              zero=sum(x == 0 for x in values), raters=len(raters)))
    args.save_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.save_dir / 'rated_cases.csv', rows, list(rows[0]))
    write_csv(args.save_dir / 'per_content.csv', per_content, list(per_content[0]))
    write_csv(args.save_dir / 'summary.csv', summaries, list(summaries[0]))
    print('Exploratory within-content correlations; each group has only four selected styles.')
    print(f'Wrote {args.save_dir}; raters: {len(raters)}')


if __name__ == '__main__':
    main()
