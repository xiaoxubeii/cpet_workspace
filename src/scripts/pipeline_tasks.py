#!/usr/bin/env python3
"""Utility helpers for CPETFormer pipeline orchestration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_values(value: str) -> List[str]:
    return [item for item in value.split() if item]


def cmd_write_info(args: argparse.Namespace) -> None:
    pipeline_root = Path(args.pipeline_root)
    step_dir = Path(args.step_dir)
    pipeline_root.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = pipeline_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "pipeline_name": args.pipeline_name,
        "model": args.model,
        "baseline_arch": args.baseline_arch,
        "subset_data": args.subset_data,
        "full_data": args.full_data,
        "timestamp": _now_iso(),
        "lr_grid": _split_values(args.lr_grid),
        "wd_grid": _split_values(args.wd_grid),
        "grad_clip_grid": _split_values(args.grad_clip_grid),
        "sprint_epochs": int(args.sprint_epochs),
        "full_epochs": int(args.full_epochs),
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        "pipeline_parent": str(pipeline_root.parent),
    }

    summary_path = step_dir / "pipeline_info.json"
    summary_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_write_meta(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "arch": args.arch,
        "combo": args.combo,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "stage": args.stage,
    }
    (run_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_copy_meta(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    run_root = Path(args.run_root)
    run_root.parent.mkdir(parents=True, exist_ok=True)
    source_meta = source_dir / "meta.json"
    if source_meta.is_file():
        run_meta = run_root / "meta.json"
        run_meta.write_text(source_meta.read_text(encoding="utf-8"), encoding="utf-8")


def _load_eval_entries(meta_paths: Sequence[Path]) -> List[dict]:
    entries: List[dict] = []
    for meta_path in meta_paths:
        run_root = meta_path.parent
        eval_dir = run_root / "results"
        eval_file = eval_dir / "test_final_results.json"
        if not eval_file.is_file():
            nested_results = sorted(eval_dir.glob("*/all/results/test_final_results.json"))
            if nested_results:
                eval_file = nested_results[0]
            else:
                continue
        try:
            metrics = json.loads(eval_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        test_metrics = metrics.get("test_metrics", metrics)
        mae = test_metrics.get("mae")
        r2 = test_metrics.get("r2_score")
        if mae is None or r2 is None:
            continue
        rmse = test_metrics.get("rmse")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "arch": meta.get("arch"),
                "combo": meta.get("combo"),
                "learning_rate": meta.get("learning_rate"),
                "weight_decay": meta.get("weight_decay"),
                "grad_clip": meta.get("grad_clip"),
                "mae": float(mae),
                "r2_score": float(r2),
                "rmse": float(rmse) if rmse is not None else None,
                "eval_path": str(eval_file.parent),
                "timestamp": _now_iso(),
            }
        )
    return entries


def cmd_summarize_sprint(args: argparse.Namespace) -> None:
    pipeline_root = Path(args.pipeline_root)
    step_dir = Path(args.step_dir)
    reports_dir = pipeline_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "results").mkdir(parents=True, exist_ok=True)
    (step_dir / "reports").mkdir(parents=True, exist_ok=True)

    meta_paths = sorted((pipeline_root / "sprint-eval").glob("*/meta.json"))
    entries = _load_eval_entries(meta_paths)
    if not entries:
        print("[sprint-summarize] No evaluation results found. Using training metadata as fallback.")
        train_meta_paths = sorted((pipeline_root / "sprint-train").glob("*/meta.json"))
        for meta_path in train_meta_paths:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entries.append(
                {
                    "arch": meta.get("arch"),
                    "combo": meta.get("combo"),
                    "learning_rate": meta.get("learning_rate"),
                    "weight_decay": meta.get("weight_decay"),
                    "grad_clip": meta.get("grad_clip"),
                    "mae": 9_999.0,
                    "r2_score": -9_999.0,
                    "rmse": None,
                    "eval_path": "",
                    "timestamp": _now_iso(),
                }
            )
        if not entries:
            return

    entries.sort(key=lambda x: (x["mae"], -x["r2_score"]))
    summary = {
        "stage": "sprint",
        "generated_at": _now_iso(),
        "entries": entries,
    }
    (step_dir / "results" / "sprint_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    top_k = max(1, int(args.top_k))
    top_entries = entries[:top_k]
    top_yaml = {
        "stage": "sprint",
        "selected": [
            {
                "arch": entry["arch"],
                "combo": entry["combo"],
                "learning_rate": entry["learning_rate"],
                "weight_decay": entry["weight_decay"],
                "grad_clip": entry["grad_clip"],
            }
            for entry in top_entries
        ],
    }
    with open(step_dir / "reports" / "top_full_candidates.yaml", "w", encoding="utf-8") as f:
        yaml.dump(top_yaml, f, sort_keys=False)
    (reports_dir / "top_full_candidates.json").write_text(
        json.dumps(top_yaml, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(reports_dir / "top_full_candidates.txt", "w", encoding="utf-8") as f:
        for entry in top_entries:
            f.write(
                "{arch}\t{lr}\t{wd}\t{clip}\n".format(
                    arch=entry["arch"],
                    lr=entry["learning_rate"],
                    wd=entry["weight_decay"],
                    clip=entry["grad_clip"],
                )
            )

    md_lines = ["# Sprint Results", "", "| Rank | Combo | MAE | R² |", "|---|---|---|---|"]
    for idx, entry in enumerate(top_entries, start=1):
        md_lines.append(f"| {idx} | {entry['combo']} | {entry['mae']:.4f} | {entry['r2_score']:.4f} |")
    (step_dir / "reports" / "sprint_summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def cmd_emit_candidates(args: argparse.Namespace) -> None:
    path = Path(args.pipeline_root) / "reports" / "top_full_candidates.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in data.get("selected", []):
            f.write(
                "\t".join(
                    [
                        entry.get("arch", args.default_arch),
                        str(entry.get("combo", "")),
                        str(entry.get("learning_rate")),
                        str(entry.get("weight_decay")),
                        str(entry.get("grad_clip")),
                    ]
                )
                + "\n"
            )


def cmd_summarize_full(args: argparse.Namespace) -> None:
    pipeline_root = Path(args.pipeline_root)
    step_dir = Path(args.step_dir)
    reports_dir = pipeline_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "results").mkdir(parents=True, exist_ok=True)
    (step_dir / "reports").mkdir(parents=True, exist_ok=True)

    meta_paths = sorted((pipeline_root / "full-eval").glob("*/meta.json"))
    entries = _load_eval_entries(meta_paths)
    if not entries:
        print("[full-summarize] No evaluation results found. Using training metadata as fallback.")
        train_meta_paths = sorted((pipeline_root / "full-train").glob("*/meta.json"))
        for meta_path in train_meta_paths:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entries.append(
                {
                    "arch": meta.get("arch"),
                    "combo": meta.get("combo"),
                    "learning_rate": meta.get("learning_rate"),
                    "weight_decay": meta.get("weight_decay"),
                    "grad_clip": meta.get("grad_clip"),
                    "mae": 9_999.0,
                    "r2_score": -9_999.0,
                    "rmse": None,
                    "eval_path": "",
                }
            )
        if not entries:
            return

    for entry in entries:
        loco_path = Path(entry["eval_path"]).parent / "loco" / "aggregate_results.json"
        if loco_path.is_file():
            try:
                entry["loco_metrics"] = json.loads(loco_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    entries.sort(key=lambda x: (x["mae"], -x["r2_score"]))
    summary = {
        "stage": "full",
        "generated_at": _now_iso(),
        "entries": entries,
    }
    (step_dir / "results" / "full_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    best = entries[0]
    best_payload = {
        "arch": best["arch"],
        "combo": best["combo"],
        "learning_rate": best["learning_rate"],
        "weight_decay": best["weight_decay"],
        "grad_clip": best["grad_clip"],
        "mae": best["mae"],
        "r2_score": best["r2_score"],
        "rmse": best.get("rmse"),
        "eval_path": best["eval_path"],
        "generated_at": summary["generated_at"],
    }
    (reports_dir / "best_model.json").write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = ["# Full Training Results", "", "| Rank | Combo | MAE | R² |", "|---|---|---|---|"]
    for idx, entry in enumerate(entries[:5], start=1):
        md_lines.append(f"| {idx} | {entry['combo']} | {entry['mae']:.4f} | {entry['r2_score']:.4f} |")
    (step_dir / "reports" / "full_summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def cmd_baseline_report(args: argparse.Namespace) -> None:
    pipeline_root = Path(args.pipeline_root)
    reports_dir = pipeline_root / "reports"
    step_dir = Path(args.step_dir)
    step_reports = step_dir / "reports"
    step_reports.mkdir(parents=True, exist_ok=True)

    best_path = reports_dir / "best_model.json"
    baseline_eval = (
        pipeline_root / "baseline-eval" / "runs" / args.baseline_arch / "eval" / "results" / "test_final_results.json"
    )
    if not best_path.is_file() or not baseline_eval.is_file():
        print("[baseline-report] Missing best model or baseline metrics.")
        return

    best = json.loads(best_path.read_text(encoding="utf-8"))
    baseline_data = json.loads(baseline_eval.read_text(encoding="utf-8"))
    baseline_metrics = baseline_data.get("test_metrics", baseline_data)

    report = {
        "generated_at": _now_iso(),
        "best_model": best,
        "baseline_arch": args.baseline_arch,
        "baseline_metrics": baseline_metrics,
    }
    (step_reports / "baseline_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def _fmt(value):
        return "N/A" if value is None else f"{value:.4f}"

    md_lines = [
        "# Baseline Comparison",
        "",
        f"- Tuned model: **{best['arch']}** ({best['combo']})",
        f"- Baseline: **{args.baseline_arch}**",
        "",
        "| Metric | Tuned | Baseline |",
        "|---|---|---|",
    ]
    for metric in ["mae", "r2_score", "rmse"]:
        md_lines.append(f"| {metric.upper()} | {_fmt(best.get(metric))} | {_fmt(baseline_metrics.get(metric))} |")
    (step_reports / "baseline_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")

    for filename in ["baseline_comparison.json", "baseline_comparison.md"]:
        src = step_reports / filename
        dst = reports_dir / filename
        dst.write_bytes(src.read_bytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPETFormer pipeline helper tasks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-info", help="Write pipeline metadata and update symlink.")
    p.add_argument("--pipeline-root", required=True)
    p.add_argument("--step-dir", required=True)
    p.add_argument("--pipeline-name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--baseline-arch", required=True)
    p.add_argument("--subset-data", required=True)
    p.add_argument("--full-data", required=True)
    p.add_argument("--lr-grid", required=True)
    p.add_argument("--wd-grid", required=True)
    p.add_argument("--grad-clip-grid", required=True)
    p.add_argument("--sprint-epochs", required=True)
    p.add_argument("--full-epochs", required=True)
    p.add_argument("--batch-size", required=True)
    p.add_argument("--top-k", required=True)
    p.set_defaults(func=cmd_write_info)

    p = sub.add_parser("write-meta", help="Persist combination metadata.")
    p.add_argument("--run-root", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--combo", required=True)
    p.add_argument("--learning-rate", required=True)
    p.add_argument("--weight-decay", required=True)
    p.add_argument("--grad-clip", required=True)
    p.add_argument("--stage", required=True)
    p.set_defaults(func=cmd_write_meta)

    p = sub.add_parser("copy-meta", help="Copy combination metadata between stages.")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--run-root", required=True)
    p.set_defaults(func=cmd_copy_meta)

    p = sub.add_parser("summarize-sprint", help="Aggregate sprint evaluation metrics.")
    p.add_argument("--pipeline-root", required=True)
    p.add_argument("--step-dir", required=True)
    p.add_argument("--top-k", required=True)
    p.set_defaults(func=cmd_summarize_sprint)

    p = sub.add_parser("emit-candidates", help="Emit top combinations for full training.")
    p.add_argument("--pipeline-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--default-arch", required=True)
    p.set_defaults(func=cmd_emit_candidates)

    p = sub.add_parser("summarize-full", help="Aggregate full evaluation metrics.")
    p.add_argument("--pipeline-root", required=True)
    p.add_argument("--step-dir", required=True)
    p.set_defaults(func=cmd_summarize_full)

    p = sub.add_parser("baseline-report", help="Generate baseline comparison report.")
    p.add_argument("--pipeline-root", required=True)
    p.add_argument("--step-dir", required=True)
    p.add_argument("--baseline-arch", required=True)
    p.set_defaults(func=cmd_baseline_report)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
