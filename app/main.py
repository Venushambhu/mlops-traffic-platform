"""FastAPI service exposing the trained traffic signal control policies.

Probe design (this is what Kubernetes depends on):

  /health  - liveness.  Answers as soon as the process is up. Failing this means
             the process is wedged, and the kubelet should restart it.
  /ready   - readiness. Returns 503 until every agent's weights are in memory.
             Failing this removes the pod from Service endpoints but does NOT
             restart it.

Collapsing these is the classic mistake: models take tens of seconds to load,
so a combined endpoint under an aggressive liveness probe produces a permanent
restart loop — the pod never lives long enough to finish loading.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.config import settings
from app.model import policy_service
from app.schemas import (
    HealthResponse,
    PredictionResponse,
    ReadinessResponse,
    TrafficState,
)

logging.basicConfig(
    level=settings.log_level,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

# --- metrics -------------------------------------------------------------
# Infrastructure signals (latency, errors) plus model signals (action
# distribution, input feature ranges, gating rate). The model signals are what
# make this observable as an ML service rather than just an HTTP service.

PREDICTIONS = Counter(
    "tsc_predictions_total", "Prediction requests served", ["agent", "action_source"]
)
REQUESTED_ACTION = Counter(
    "tsc_requested_action_total", "What the policy asked for", ["agent", "action"]
)
GATED = Counter(
    "tsc_gated_by_min_green_total",
    "Switch requests downgraded to hold because minimum green had not elapsed",
    ["agent"],
)
INFERENCE_LATENCY = Histogram(
    "tsc_inference_duration_seconds",
    "Model inference latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
FEATURE = Histogram(
    "tsc_input_feature",
    "Normalised input feature values (drift signal against the training range)",
    ["feature"],
    buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)
CLIPPED = Counter(
    "tsc_input_clipped_total",
    "Inputs that hit the normalisation ceiling — saturation, possible drift",
    ["feature"],
)
MODEL_READY = Gauge("tsc_model_ready", "1 when every expected agent is loaded")

FEATURE_NAMES = (
    "queue",
    "waiting",
    "occupancy",
    "phase",
    "green_elapsed",
    "neighbour_queue",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting %s", settings.app_name)
    policy_service.load()
    MODEL_READY.set(1 if policy_service.is_ready else 0)
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Traffic Signal Policy Service",
    description=(
        "Serves independent DQN policies for adaptive traffic signal control, "
        "one per intersection."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["probes"])
def health() -> HealthResponse:
    """Liveness: the process is running and can serve HTTP."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadinessResponse, tags=["probes"])
def ready(response: Response) -> ReadinessResponse:
    """Readiness: every expected agent is loaded and this pod can take traffic."""
    if not policy_service.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        missing = sorted(set(settings.agents) - set(policy_service.loaded_agents))
        return ReadinessResponse(
            ready=False,
            agents_loaded=policy_service.loaded_agents,
            agents_expected=settings.agents,
            detail=policy_service.load_error or f"still loading; missing {missing}",
        )
    return ReadinessResponse(
        ready=True,
        agents_loaded=policy_service.loaded_agents,
        agents_expected=settings.agents,
        model_version=policy_service.version,
    )


@app.get("/metrics", tags=["probes"])
def metrics() -> Response:
    MODEL_READY.set(1 if policy_service.is_ready else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(state: TrafficState, response: Response) -> PredictionResponse:
    if not policy_service.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return PredictionResponse(
            intersection_id=state.intersection_id,
            requested_action=0,
            applied_action=0,
            action_label="hold",
            gated_by_min_green=False,
            action_source="fallback",
            model_version="unloaded",
            observation=[],
            inference_ms=0.0,
        )

    if not policy_service.has_agent(state.intersection_id):
        # A request for an intersection this deployment does not serve is a
        # client error, not a server failure.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no policy for intersection '{state.intersection_id}'; "
                f"served: {policy_service.loaded_agents}"
            ),
        )

    observation = policy_service.build_observation(
        queue_length=state.queue_length,
        waiting_time=state.waiting_time,
        occupancy=state.occupancy,
        current_phase=state.current_phase,
        num_phases=state.num_phases,
        green_elapsed=state.green_elapsed,
        neighbour_queue_total=state.neighbour_queue_total,
        num_neighbours=state.num_neighbours,
    )

    for name, value in zip(FEATURE_NAMES, observation):
        FEATURE.labels(feature=name).observe(float(value))
        if value >= 1.0:
            CLIPPED.labels(feature=name).inc()

    started = time.perf_counter()
    with INFERENCE_LATENCY.time():
        requested, source = policy_service.predict(state.intersection_id, observation)
    elapsed_ms = (time.perf_counter() - started) * 1000

    applied, gated = policy_service.apply_control_constraints(
        requested, state.green_elapsed
    )

    PREDICTIONS.labels(agent=state.intersection_id, action_source=source).inc()
    REQUESTED_ACTION.labels(agent=state.intersection_id, action=str(requested)).inc()
    if gated:
        GATED.labels(agent=state.intersection_id).inc()

    return PredictionResponse(
        intersection_id=state.intersection_id,
        requested_action=requested,
        applied_action=applied,
        action_label="switch" if applied == 1 else "hold",
        gated_by_min_green=gated,
        action_source=source,
        model_version=policy_service.version or "unknown",
        observation=[round(float(v), 4) for v in observation],
        inference_ms=round(elapsed_ms, 3),
    )
