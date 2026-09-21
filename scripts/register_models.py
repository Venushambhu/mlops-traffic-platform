"""Log the thesis DQN checkpoints into MLflow tracking and the model registry.

These models were trained during my M.Sc. thesis, outside MLflow. This script
registers the existing artifacts and logs their real training configuration and
evaluation results retrospectively, read from the thesis output files rather
than typed in by hand. In a production pipeline the training job would log to
MLflow directly; this is the bridge for models that already exist.

One MLflow run per agent, because the thesis trains independent learners — four
separate policies, four separate artifacts, four registered models.

Usage:
    python scripts/register_models.py
    python scripts/register_models.py --scenario low --promote-to Staging
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

REPO = Path(__file__).resolve().parent.parent
WRAPPER = Path(__file__).resolve().parent / "sb3_wrapper.py"
EVAL_CSV = REPO / "data" / "eval_summary_2x2_demand.csv"
COMPUTE_CSV = REPO / "data" / "training_compute_master.csv"

# Hyperparameters that matter for reproducing a run. Read from the checkpoint
# itself rather than a config file, so they cannot drift from the artifact.
HYPERPARAMS = (
    "learning_rate",
    "gamma",
    "batch_size",
    "buffer_size",
    "learning_starts",
    "target_update_interval",
    "exploration_fraction",
    "exploration_initial_eps",
    "exploration_final_eps",
    "gradient_steps",
    "tau",
)


def read_checkpoint_params(path: Path) -> dict:
    """Pull hyperparameters and the training environment out of the .zip."""
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("data"))
        sb3_version = z.read("_stable_baselines3_version").decode().strip()
        system_info = z.read("system_info.txt").decode()

    params = {k: data[k] for k in HYPERPARAMS if k in data and not isinstance(data[k], dict)}
    params["sb3_version"] = sb3_version

    # system_info.txt lines look like "- PyTorch: 2.13.0"
    for line in system_info.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.lstrip("- ").partition(":")
        key = key.strip().lower().replace("-", "_").replace(" ", "_")
        if key in {"python", "pytorch", "numpy", "gymnasium"}:
            params[f"train_env_{key}"] = value.strip()

    return params


def read_eval_metrics(grid: str, scenario: str, controller: str = "dqn_v2") -> dict:
    """Read real evaluation results from the thesis analysis output.

    Metrics come from the evaluation artifact, not from anything typed in here.
    These are means over 5 seeds, as reported in the thesis.
    """
    if not EVAL_CSV.exists():
        print(f"  ! {EVAL_CSV.name} missing — skipping evaluation metrics")
        return {}

    with EVAL_CSV.open() as f:
        for row in csv.DictReader(f):
            if (
                row["grid"] == grid
                and row["scenario"] == scenario
                and row["controller"] == controller
            ):
                return {
                    "eval_completion_rate_pct": float(row["completion_rate_pct_mean"]),
                    "eval_avg_travel_time_s": float(row["avg_travel_time_mean"]),
                    "eval_avg_waiting_time_s": float(row["avg_waiting_time_mean"]),
                    "eval_avg_time_loss_s": float(row["avg_time_loss_mean"]),
                    "eval_mean_queue_length": float(row["mean_queue_length_mean"]),
                    "eval_seeds": int(row["n"]),
                }
    print(f"  ! no eval row for {grid}/{scenario}/{controller}")
    return {}


def read_training_compute(grid: str, scenario: str, algorithm: str = "dqn") -> dict:
    """Read what the training run actually cost — steps, runtime, memory."""
    if not COMPUTE_CSV.exists():
        return {}

    with COMPUTE_CSV.open() as f:
        for row in csv.DictReader(f):
            if (
                row["algorithm"] == algorithm
                and row["grid"] == grid
                and row["scenario"] == scenario
            ):
                return {
                    "train_total_steps": int(row["total_steps"]),
                    "train_episodes": int(row["episodes_completed"]),
                    "train_runtime_s": float(row["training_runtime_s"]),
                    "train_peak_ram_mb": float(row["peak_process_tree_ram_mb"]),
                }
    return {}


def register_agent(
    agent: str,
    checkpoint: Path,
    grid: str,
    scenario: str,
    experiment: str,
    promote_to: str | None,
) -> None:
    model_name = f"tsc-dqn-{grid}-{scenario}-{agent}"
    print(f"\n{agent}  ->  {model_name}")

    params = read_checkpoint_params(checkpoint)
    metrics = {**read_eval_metrics(grid, scenario), **read_training_compute(grid, scenario)}

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{grid}-{scenario}-{agent}") as run:
        mlflow.log_params(
            {
                **params,
                "agent": agent,
                "grid": grid,
                "scenario": scenario,
                "algorithm": "DQN",
                "marl_setting": "independent_learners",
                "obs_dim": 6,
                "action_space": "Discrete(2)",
                "min_green_seconds": 10,
                "decision_interval_seconds": 5,
            }
        )
        if metrics:
            mlflow.log_metrics(metrics)

        # Tags describe provenance: these models predate this pipeline, and
        # saying so in the metadata is better than leaving it ambiguous.
        mlflow.set_tags(
            {
                "source": "msc-thesis-revision-v2",
                "logged_retrospectively": "true",
                "checkpoint_file": checkpoint.name,
            }
        )

        mlflow.pyfunc.log_model(
            name="model",
            python_model=str(WRAPPER),
            artifacts={"checkpoint": str(checkpoint)},
            registered_model_name=model_name,
            pip_requirements=[
                f"stable-baselines3=={params.get('sb3_version', '2.9.0')}",
                "torch",
                "numpy",
            ],
        )
        print(f"  run_id={run.info.run_id}")
        if metrics:
            print(f"  logged {len(metrics)} metrics, {len(params) + 9} params")

    if promote_to:
        client = MlflowClient()
        version = max(int(mv.version) for mv in client.search_model_versions(f"name='{model_name}'"))
        client.set_registered_model_alias(model_name, promote_to, version)
        print(f"  version {version} -> alias '{promote_to}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="2x2")
    ap.add_argument("--scenario", default="medium")
    ap.add_argument("--agents", default="B1,B2,C1,C2")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--experiment", default="traffic-signal-control")
    ap.add_argument(
        "--promote-to",
        default=None,
        help="Alias to assign, e.g. Staging or Production. Omit to register only.",
    )
    args = ap.parse_args()

    model_dir = Path(
        args.model_dir or REPO / f"models/grid{args.grid}/dqn/{args.scenario}"
    )
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    print(f"registering {len(agents)} agents from {model_dir}")
    for agent in agents:
        checkpoint = model_dir / f"dqn_{agent}_{args.scenario}.zip"
        if not checkpoint.exists():
            print(f"\n{agent}  !! missing {checkpoint}")
            continue
        register_agent(
            agent, checkpoint, args.grid, args.scenario, args.experiment, args.promote_to
        )

    print("\ndone — view with: mlflow ui")


if __name__ == "__main__":
    main()
