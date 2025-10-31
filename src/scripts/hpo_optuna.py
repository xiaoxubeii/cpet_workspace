#!/usr/bin/env python3
"""Optuna-based hyper-parameter search for CPET pipelines.

This script wraps the existing ``cpetx-model`` CLI so it can be dropped into
the current pipeline without refactoring the training/evaluation code. Each
trial performs a short ``train`` run on the subset dataset followed by an
``eval`` on the full dataset, and the validation MAE is reported back to
Optuna. The top-k parameter sets are exported in the same format used by the
existing grid-search pipeline so downstream steps (full-train / full-eval)
continue to work unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import yaml


def _stream_run(cmd: List[str], *, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> None:
    """Execute *cmd* streaming stdout/stderr to the caller's console."""

    display = " ".join(cmd)
    print(f"[hpo] $ {display}")
    process = subprocess.Popen(  # noqa: S603,S607
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with return code {return_code}: {display}")


def _extract_metrics(eval_root: Path) -> Dict[str, float]:
    """Return scalar metrics (MAE/RMSE/R²) from the evaluation directory."""

    metrics_path = eval_root / "results" / "test_final_results.json"
    if not metrics_path.is_file():
        nested = sorted((eval_root / "results").glob("*/all/results/test_final_results.json"))
        if nested:
            metrics_path = nested[0]
    if not metrics_path.is_file():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    container = data.get("test_metrics", data)
    result: Dict[str, float] = {}
    for key in ("mae", "rmse", "r2_score"):
        value = container.get(key)
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result


def _normalise_arch_pattern(pattern: str) -> str:
    """Convert a regex like ``^cpet_former$`` into ``cpet_former`` for reporting."""

    stripped = pattern.strip()
    if stripped.startswith("^"):
        stripped = stripped[1:]
    if stripped.endswith("$"):
        stripped = stripped[:-1]
    return stripped


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyper-parameter optimisation helper")
    parser.add_argument("--arch", required=True, help="Regex for target architecture (e.g. ^cpet_former$")
    parser.add_argument("--subset-data", required=True, help="Path to subset dataset for sprint training")
    parser.add_argument("--full-data", required=True, help="Path to full dataset for evaluation")
    parser.add_argument("--configs-dir", required=True, help="Directory containing model/ and eval/ configs")
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory (root for this model)")
    parser.add_argument("--sprint-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study", default="cpet_optuna")
    parser.add_argument("--storage", default=None, help="Optuna storage URI (default: sqlite file under run dir)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    reports_dir = run_dir / "reports"
    optuna_dir = run_dir / "optuna"
    reports_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir.mkdir(parents=True, exist_ok=True)

    if args.storage:
        storage = args.storage
    else:
        storage = f"sqlite:///{(optuna_dir / 'optuna.db').as_posix()}"

    configs_dir = Path(args.configs_dir).expanduser().resolve()
    model_conf = configs_dir / "model"
    eval_conf = configs_dir / "eval"

    project_root = Path(os.environ.get("CPETX_PROJECT_ROOT", run_dir.parent)).resolve()
    python_exec = "python"
    cpetx_model = project_root / "src/vox_cpet/cmd/cpetx-model"

    env_base = dict(os.environ)
    env_base.setdefault("PYTHONPATH", str(project_root / "src"))
    env_base.setdefault("CPETX_PROJECT_ROOT", str(project_root))

    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    subset_path = Path(args.subset_data).expanduser().resolve()
    full_path = Path(args.full_data).expanduser().resolve()
    arch_regex = args.arch
    arch_label = _normalise_arch_pattern(arch_regex)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-7, 5e-4, log=True)
        clip = trial.suggest_float("grad_clip", 0.3, 1.0)

        trial_root = optuna_dir / f"trial_{trial.number:04d}"
        train_root = trial_root / "train"
        eval_root = trial_root / "eval"
        for directory in (train_root / "artifacts", train_root / "logs", eval_root / "results", eval_root / "logs"):
            directory.mkdir(parents=True, exist_ok=True)

        env = dict(env_base)
        env["WEIGHT_DECAY"] = str(wd)
        env["GRAD_CLIP_MAX_NORM"] = str(clip)

        task_id = f"optuna_{trial.number}"
        combo = f"lr{lr:.6g}_wd{wd:.6g}_clip{clip:.3g}"

        _stream_run(
            [
                python_exec,
                str(cpetx_model),
                "train",
                "--data-file",
                str(subset_path),
                "--conf",
                str(model_conf),
                "--filter",
                arch_regex,
                "--run-dir",
                str(train_root),
                "--save-dir",
                str(train_root / "artifacts"),
                "--log-dir",
                str(train_root / "logs"),
                "--task-id",
                task_id,
                "--num-epochs",
                str(args.sprint_epochs),
                "--learning-rate",
                str(lr),
                "--batch-size",
                str(args.batch_size),
            ],
            env=env,
        )

        _stream_run(
            [
                python_exec,
                str(cpetx_model),
                "eval",
                "--task-id",
                task_id,
                "--conf",
                str(eval_conf),
                "--data-file",
                str(full_path),
                "--filter",
                arch_regex,
                "--run-dir",
                str(eval_root),
                "--save-dir",
                str(eval_root / "results"),
                "--log-dir",
                str(eval_root / "logs"),
                "--checkpoints-path",
                str(train_root / "artifacts"),
            ],
            env=env,
        )

        metrics = _extract_metrics(eval_root)
        mae = metrics.get("mae", float("inf"))
        trial.set_user_attr("combo", combo)
        trial.set_user_attr("metrics", metrics)
        trial.report(mae, step=1)
        return mae

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    complete_trials.sort(key=lambda t: t.value)

    leaderboard = [
        {
            "rank": idx + 1,
            "value": trial.value,
            "params": trial.params,
            "metrics": trial.user_attrs.get("metrics", {}),
            "combo": trial.user_attrs.get("combo"),
            "trial_id": trial.number,
        }
        for idx, trial in enumerate(complete_trials)
    ]
    (optuna_dir / "hpo_leaderboard.json").write_text(json.dumps(leaderboard, indent=2, ensure_ascii=False), encoding="utf-8")

    top_k = max(1, int(args.top_k))
    selected = []
    for trial in complete_trials[:top_k]:
        params = trial.params
        lr = float(params["learning_rate"])
        wd = float(params.get("weight_decay", 0.0))
        clip = float(params.get("grad_clip", 0.0))
        combo = trial.user_attrs.get("combo") or f"lr{lr:.6g}_wd{wd:.6g}_clip{clip:.3g}"
        selected.append(
            {
                "arch": arch_label,
                "combo": combo,
                "learning_rate": lr,
                "weight_decay": wd,
                "grad_clip": clip,
            }
        )

    payload = {"stage": "optuna", "selected": selected}
    json_path = reports_dir / "top_full_candidates.json"
    yaml_path = reports_dir / "top_full_candidates.yaml"
    txt_path = reports_dir / "top_full_candidates.txt"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    with txt_path.open("w", encoding="utf-8") as handle:
        for entry in selected:
            handle.write(
                f"{entry['arch']}\t{entry['combo']}\t{entry['learning_rate']}\t{entry['weight_decay']}\t{entry['grad_clip']}\n"
            )

    print(f"[hpo] Leaderboard saved to {optuna_dir/'hpo_leaderboard.json'}")
    print(f"[hpo] Top-{top_k} candidates exported to {json_path}")


if __name__ == "__main__":
    main()

