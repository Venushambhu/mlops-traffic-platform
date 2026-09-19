# Traffic Signal Policy — MLOps Platform

Production serving platform for a multi-agent reinforcement learning policy that
controls traffic signals. The model comes from my M.Sc. thesis
([Scalable MARL for adaptive traffic signal control](https://github.com/Venushambhu/Scalable-MARL-for-adaptive-traffic-signal-control-));
this repository is about everything that happens *after* a model is trained —
packaging it, deploying it, observing it, and releasing new versions of it safely.

## Why this exists

The thesis produced a working policy and a set of results. It did not produce
anything anyone could actually run. This project closes that gap: taking a model
I trained myself and putting it through the lifecycle a production system needs.

## Architecture

```
                   ┌──────────────────┐
  training run ───►│ MLflow Tracking  │
                   │  + Registry      │
                   └────────┬─────────┘
                            │ models:/tsc-dqn/Production
                            ▼
  request ──► ┌─────────────────────────────┐ ──► recommended phase
              │  FastAPI inference service  │
              │  /predict /health /ready    │
              │  /metrics                   │
              └──────────┬──────────────────┘
                         │ scraped
                         ▼
                 Prometheus ──► Grafana
```

Runtime: Docker image → Kubernetes Deployment (multi-replica, probes, resource
limits) → promoted staging-to-production by a GitLab CI pipeline. AWS resources
(S3 artifact store, ECR, IAM roles) provisioned with Terraform.

## Service contract

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Traffic state in, recommended signal phase out |
| `GET /health` | **Liveness** — process is up. Succeeds even if the model is unloaded |
| `GET /ready` | **Readiness** — 503 until weights are in memory; reports load errors |
| `GET /metrics` | Prometheus exposition — latency, errors, phase distribution, input ranges |

`/health` and `/ready` are separate on purpose. A model that takes tens of
seconds to load will fail an aggressive liveness probe, get killed by the
kubelet, and restart forever. Readiness removes the pod from the Service while
it loads; liveness only restarts a process that is genuinely wedged.

### Example

```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "intersection_id": "B1",
        "queue_lengths": [4.0, 2.0, 7.0, 1.0],
        "current_phase": 1,
        "time_in_phase": 12.5
      }'
```

```json
{
  "intersection_id": "B1",
  "recommended_phase": 2,
  "action_source": "policy",
  "model_version": "tsc-dqn@Production",
  "inference_ms": 1.84
}
```

`action_source` is `fallback` when the policy could not be applied. A signal
controller that loses its optimiser should keep running a valid plan rather than
stop responding, so inference failures degrade to a safe default phase instead
of returning a 500.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# place a trained checkpoint at models/dqn_grid2x2.zip, then:
uvicorn app.main:app --reload

pytest -q
```

Configuration is environment-driven with the `TSC_` prefix
(`TSC_MODEL_PATH`, `TSC_MLFLOW_MODEL_NAME`, `TSC_MODEL_LOAD_DELAY_SECONDS`, …).
Nothing is hardcoded to an environment — that is what makes the same image
promotable from staging to production unchanged.

## Layout

```
app/
  schemas.py   validated request/response contract
  config.py    environment-driven settings
  model.py     model loading + inference (the only file that knows where models come from)
  main.py      routes, probes, metrics
tests/         contract tests using an injected stub policy
```

`model.py` is isolated so that switching from a local checkpoint to the MLflow
Model Registry touches one file and nothing else.

## Status

- [x] Inference service, validated contract, probes, metrics, tests
- [ ] MLflow tracking + registry, stage-based model resolution
- [ ] Docker multi-stage image
- [ ] Kubernetes deployment (probes, limits, rolling updates)
- [ ] GitLab CI — build once, promote the same artifact
- [ ] Terraform — S3, ECR, least-privilege IAM
- [ ] Prometheus + Grafana dashboards, drift signals
- [ ] Secrets via AWS Secrets Manager, image scanning

Design decisions and their trade-offs are recorded in [DECISIONS.md](DECISIONS.md).
