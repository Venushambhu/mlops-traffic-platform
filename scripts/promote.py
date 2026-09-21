"""Point an alias at a specific model version, for all agents at once.

Usage:
    python scripts/promote.py --version 1
    python scripts/promote.py --version 2 --alias Staging
"""
import argparse
from mlflow.tracking import MlflowClient

ap = argparse.ArgumentParser()
ap.add_argument("--version", required=True)
ap.add_argument("--alias", default="Production")
ap.add_argument("--prefix", default="tsc-dqn-2x2-medium")
ap.add_argument("--agents", default="B1,B2,C1,C2")
args = ap.parse_args()

client = MlflowClient()
for agent in args.agents.split(","):
    name = f"{args.prefix}-{agent}"
    client.set_registered_model_alias(name, args.alias, args.version)
    print(f"{name}: {args.alias} -> v{args.version}")