#!/usr/bin/env python3
"""
Convert legacy baseline summary.json files into collect-reports/reports artifacts.

The generated files mimic the layout produced by the multi-model HPO pipeline so
that downstream tooling (e.g. rank-model) can consume historical baselines.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Candidate:
    arch: str
    combo: str
    mae: Optional[float]
    rmse: Optional[float]
    r2_score: Optional[float]
    correlation: Optional[float]
    mape: Optional[float]
    learning_rate: Optional[float]
    weight_decay: Optional[float]
    grad_clip: Optional[float]
    eval_path: Optional[str]
    generated_at: Optional[str]
    source_summary: str
    pipeline_dir: str


def _maybe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_tuple(candidate: Candidate) -> Tuple[float, float, float]:
    def to_float(value: Optional[float]) -> float:
        return float(value) if value is not None else float("inf")

    mae = to_float(candidate.mae)
    rmse = to_float(candidate.rmse)
    r2_component = -to_float(candidate.r2_score)
    return (mae, rmse, r2_component)


def _load_candidates(summary_path: Path) -> Iterable[Candidate]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    run_info = data.get("run_info") or {}
    default_timestamp = run_info.get("summary_time")
    pipeline_dir = str(summary_path.parent)

    models: Dict[str, Dict[str, dict]] = data.get("models") or {}
    for arch, variants in models.items():
        variants = variants or {}
        for combo_name, payload in variants.items():
            metrics_block = payload.get("metrics") or {}
            test_metrics = metrics_block.get("test_metrics") or {}
            timestamp = metrics_block.get("timestamp") or default_timestamp

            yield Candidate(
                arch=arch,
                combo=combo_name or arch,
                mae=_maybe_float(test_metrics.get("mae")),
                rmse=_maybe_float(test_metrics.get("rmse")),
                r2_score=_maybe_float(test_metrics.get("r2_score")),
                correlation=_maybe_float(test_metrics.get("correlation")),
                mape=_maybe_float(test_metrics.get("mape")),
                learning_rate=_maybe_float(payload.get("learning_rate")),
                weight_decay=_maybe_float(payload.get("weight_decay")),
                grad_clip=_maybe_float(payload.get("grad_clip")),
                eval_path=payload.get("model_path"),
                generated_at=timestamp,
                source_summary=str(summary_path),
                pipeline_dir=pipeline_dir,
            )


def _write_best_model(out_dir: Path, candidate: Candidate) -> Path:
    payload = {
        "arch": candidate.arch,
        "combo": candidate.combo,
        "learning_rate": candidate.learning_rate,
        "weight_decay": candidate.weight_decay,
        "grad_clip": candidate.grad_clip,
        "mae": candidate.mae,
        "r2_score": candidate.r2_score,
        "rmse": candidate.rmse,
        "correlation": candidate.correlation,
        "mape": candidate.mape,
        "eval_path": candidate.eval_path,
        "generated_at": candidate.generated_at or datetime.utcnow().isoformat() + "Z",
    }
    best_path = out_dir / f"{candidate.arch}_best_model.json"
    best_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return best_path


def _write_top_candidates(out_dir: Path, arch: str, candidates: List[Candidate]) -> Path:
    selected = [
        {
            "arch": c.arch,
            "combo": c.combo,
            "learning_rate": c.learning_rate,
            "weight_decay": c.weight_decay,
            "grad_clip": c.grad_clip,
        }
        for c in candidates
    ]
    payload = {
        "stage": "baseline",
        "selected": selected,
    }
    top_path = out_dir / f"{arch}_top_full_candidates.json"
    top_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return top_path


def convert_baseline_reports(baseline_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_entries = []
    for summary_path in sorted(baseline_dir.glob("*/summary.json")):
        candidates_by_arch: Dict[str, List[Candidate]] = {}
        for candidate in _load_candidates(summary_path):
            candidates_by_arch.setdefault(candidate.arch, []).append(candidate)

        for arch, candidates in candidates_by_arch.items():
            candidates.sort(key=_score_tuple)
            best_candidate = candidates[0]

            best_path = _write_best_model(output_dir, best_candidate)
            top_path = _write_top_candidates(output_dir, arch, candidates)

            summary_entries.append(
                {
                    "model": arch,
                    "pipeline_dir": best_candidate.pipeline_dir,
                    "top_full_candidates.json": str(top_path),
                    "best_model.json": str(best_path),
                }
            )

    summary_payload = {"models": summary_entries}
    summary_path = output_dir / "multi_model_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert baseline model summaries to collect-reports format."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "sota" / "baseline",
        help="Directory containing legacy baseline runs (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to <baseline-dir>/collect-reports/reports.",
    )
    args = parser.parse_args()

    baseline_dir: Path = args.baseline_dir
    if not baseline_dir.exists():
        raise SystemExit(f"Baseline directory not found: {baseline_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = baseline_dir / "collect-reports" / "reports"

    convert_baseline_reports(baseline_dir, output_dir)


if __name__ == "__main__":
    main()
