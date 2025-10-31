#!/usr/bin/env python3
"""Aggregate LOCO metrics across dataset runs.

Supports both evaluation directories (containing results/test_final_results.json)
and training run directories (containing summary.json)."""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, Optional


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_from_summary(summary_path: str) -> Optional[Dict[str, float]]:
    data = _load_json(summary_path)
    models = data.get("models", {})
    for model_entry in models.values():
        if not isinstance(model_entry, dict):
            continue
        for experiment in model_entry.values():
            metrics = experiment.get("metrics", {})
            test_metrics = metrics.get("test_metrics")
            if isinstance(test_metrics, dict):
                return {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
    return None


def _extract_from_results(results_path: str) -> Optional[Dict[str, float]]:
    data = _load_json(results_path)
    test_metrics = data.get("test_metrics", data)
    if isinstance(test_metrics, dict):
        return {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
    return None


def _collect_dataset_metrics(dataset_dir: str) -> Optional[Dict[str, float]]:
    summary_path = os.path.join(dataset_dir, "summary.json")
    if os.path.isfile(summary_path):
        metrics = _extract_from_summary(summary_path)
        if metrics:
            return metrics

    candidates = [
        os.path.join(dataset_dir, "results", "test_final_results.json"),
        os.path.join(dataset_dir, "artifacts", "results", "test_final_results.json"),
        os.path.join(dataset_dir, "artifacts", "cpa_former", "results", "test_final_results.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            metrics = _extract_from_results(path)
            if metrics:
                return metrics
    return None


def main(base_dir: str, arch: str) -> None:
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    dataset_metrics: Dict[str, Dict[str, float]] = {}

    for dataset_dir in sorted(glob.glob(os.path.join(base_dir, "*"))):
        if not os.path.isdir(dataset_dir):
            continue
        metrics = _collect_dataset_metrics(dataset_dir)
        if not metrics:
            continue

        dataset_name = os.path.basename(dataset_dir)
        dataset_metrics[dataset_name] = metrics

        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value
            metric_counts[key] = metric_counts.get(key, 0) + 1

    if not metric_counts:
        print("[WARN] No metrics found to aggregate for %s" % arch)
        return

    averages = {key: metric_sums[key] / metric_counts[key] for key in metric_sums}
    aggregate = {
        "arch": arch,
        "datasets": dataset_metrics,
        "averaged_metrics": averages,
        "num_datasets": len(dataset_metrics),
    }

    out_path = os.path.join(base_dir, "aggregate_results.json")
    existing = None
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = None

    if existing == aggregate:
        print("[INFO] Aggregated metrics unchanged for %s" % arch)
        return

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print("[INFO] Aggregated metrics for %s:" % arch)
    for key, value in sorted(averages.items()):
        print("  %s: %.4f" % (key, value))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: aggregate_loco_metrics.py <base_dir> <arch>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: aggregate_loco_metrics.py <eval_base> <arch>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
