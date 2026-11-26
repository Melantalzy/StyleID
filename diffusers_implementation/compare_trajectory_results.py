"""Pair and summarize baseline versus dynamic trajectory-alignment runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


METRICS = (
    "content_structure_score",
    "content_ssim_gray",
    "content_edge_f1",
    "style_gram_similarity",
    "style_color_similarity",
    "style_patch_similarity",
    "style_patch_coverage",
    "content_score",
    "style_score",
    "balanced_score",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare paired StyleID baseline and trajectory runs"
    )
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--dynamic_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_records(directory):
    records = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".metrics.json"):
            continue
        path = os.path.join(directory, name)
        with open(path, "r", encoding="utf-8") as record_file:
            record = json.load(record_file)
        key = (
            os.path.normcase(os.path.abspath(record["content_path"])),
            os.path.normcase(os.path.abspath(record["style_path"])),
        )
        records[key] = record
    return records


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    baseline = load_records(args.baseline_dir)
    dynamic = load_records(args.dynamic_dir)
    shared_keys = sorted(set(baseline) & set(dynamic))
    if not shared_keys:
        raise ValueError("No paired content/style records were found")

    rows = []
    values = defaultdict(list)
    for key in shared_keys:
        baseline_record = baseline[key]
        dynamic_record = dynamic[key]
        row = {
            "content_path": baseline_record["content_path"],
            "style_path": baseline_record["style_path"],
            "baseline_output": baseline_record["output_path"],
            "dynamic_output": dynamic_record["output_path"],
        }
        for metric in METRICS:
            baseline_value = float(baseline_record[metric])
            dynamic_value = float(dynamic_record[metric])
            delta = dynamic_value - baseline_value
            row[f"baseline_{metric}"] = baseline_value
            row[f"dynamic_{metric}"] = dynamic_value
            row[f"delta_{metric}"] = delta
            values[f"baseline_{metric}"].append(baseline_value)
            values[f"dynamic_{metric}"].append(dynamic_value)
            values[f"delta_{metric}"].append(delta)
        rows.append(row)

    csv_path = os.path.join(args.output_dir, "paired_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"pair_count": len(rows), "metrics": {}}
    for metric in METRICS:
        baseline_values = values[f"baseline_{metric}"]
        dynamic_values = values[f"dynamic_{metric}"]
        delta_values = values[f"delta_{metric}"]
        summary["metrics"][metric] = {
            "baseline_mean": sum(baseline_values) / len(baseline_values),
            "dynamic_mean": sum(dynamic_values) / len(dynamic_values),
            "mean_delta": sum(delta_values) / len(delta_values),
            "dynamic_wins": sum(delta > 0 for delta in delta_values),
        }

    json_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=2)

    markdown_path = os.path.join(args.output_dir, "comparison_summary.md")
    with open(markdown_path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write("# StyleID trajectory alignment comparison\n\n")
        markdown_file.write(f"Paired samples: {len(rows)}\n\n")
        markdown_file.write(
            "| Metric | Baseline | Dynamic | Delta | Dynamic wins |\n"
        )
        markdown_file.write("|---|---:|---:|---:|---:|\n")
        for metric in METRICS:
            result = summary["metrics"][metric]
            markdown_file.write(
                f"| {metric} | {result['baseline_mean']:.4f} | "
                f"{result['dynamic_mean']:.4f} | {result['mean_delta']:+.4f} | "
                f"{result['dynamic_wins']}/{len(rows)} |\n"
            )

    print(f"Paired samples: {len(rows)}")
    print(f"CSV: {os.path.abspath(csv_path)}")
    print(f"JSON: {os.path.abspath(json_path)}")
    print(f"Markdown: {os.path.abspath(markdown_path)}")


if __name__ == "__main__":
    main()
