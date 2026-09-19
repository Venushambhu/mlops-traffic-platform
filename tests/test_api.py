"""API contract tests.

Tests inject stub policies rather than loading real weights: the unit under test
is the service contract, not model quality. Model quality is validated in the
training pipeline against held-out episodes.

The normalisation tests are the important ones — they assert that serving
reproduces TrafficSignalEnv._get_observation exactly. If these drift apart, the
model receives out-of-distribution input and fails silently.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.model import PolicyService, policy_service


class StubPolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def predict(self, observation: np.ndarray) -> int:
        return self.action


class BrokenPolicy:
    def predict(self, observation: np.ndarray) -> int:
        raise RuntimeError("weights corrupted")


class InvalidActionPolicy:
    def predict(self, observation: np.ndarray) -> int:
        return 7


def stubs(action: int = 1):
    return {a: StubPolicy(action) for a in settings.agents}


# Long green: min-green already satisfied, so a switch request passes through.
SWITCHABLE_STATE = {
    "intersection_id": "B1",
    "queue_length": 12,
    "waiting_time": 90.0,
    "occupancy": 0.4,
    "current_phase": 0,
    "num_phases": 4,
    "green_elapsed": 25.0,
    "neighbour_queue_total": 8,
    "num_neighbours": 2,
}

# Same, but green has only just started.
FRESH_GREEN_STATE = {**SWITCHABLE_STATE, "green_elapsed": 4.0}


@pytest.fixture
def client():
    policy_service.set_policies(stubs(1), version="stub:v1")
    with TestClient(app) as c:
        # TestClient runs lifespan, which attempts a real load; re-inject after.
        policy_service.set_policies(stubs(1), version="stub:v1")
        yield c


# --- probes ---------------------------------------------------------------


def test_health_is_up_regardless_of_model(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ready_lists_all_expected_agents(client):
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert body["agents_loaded"] == sorted(settings.agents)


def test_not_ready_when_an_agent_is_missing(client):
    partial = {a: StubPolicy(0) for a in settings.agents[:-1]}
    policy_service.set_policies(partial, version="partial")
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


# --- normalisation: must match TrafficSignalEnv._get_observation ----------


def test_observation_matches_training_normalisation():
    obs = PolicyService.build_observation(
        queue_length=10,
        waiting_time=100.0,
        occupancy=0.5,
        current_phase=1,
        num_phases=4,
        green_elapsed=30.0,
        neighbour_queue_total=12,
        num_neighbours=2,
    )
    expected = np.array(
        [10 / 20, 100 / 200, 0.5, 1 / 3, 30 / 60, (12 / 2) / 10], dtype=np.float32
    )
    np.testing.assert_allclose(obs, expected, rtol=1e-6)


def test_every_feature_is_clipped_to_the_training_range():
    obs = PolicyService.build_observation(
        queue_length=999,
        waiting_time=9999.0,
        occupancy=1.0,
        current_phase=3,
        num_phases=4,
        green_elapsed=600.0,
        neighbour_queue_total=999,
        num_neighbours=1,
    )
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert (obs <= 1.0).all() and (obs >= 0.0).all()


def test_single_phase_programme_does_not_divide_by_zero():
    obs = PolicyService.build_observation(0, 0, 0, 0, 1, 0, 0, 0)
    assert obs[3] == 0.0


# --- control constraints --------------------------------------------------


def test_switch_is_gated_before_minimum_green(client):
    r = client.post("/predict", json=FRESH_GREEN_STATE)
    body = r.json()
    assert body["requested_action"] == 1
    assert body["applied_action"] == 0
    assert body["gated_by_min_green"] is True
    assert body["action_label"] == "hold"


def test_switch_passes_after_minimum_green(client):
    body = client.post("/predict", json=SWITCHABLE_STATE).json()
    assert body["requested_action"] == 1
    assert body["applied_action"] == 1
    assert body["gated_by_min_green"] is False
    assert body["action_label"] == "switch"


def test_hold_is_never_gated(client):
    policy_service.set_policies(stubs(0), version="stub:hold")
    body = client.post("/predict", json=FRESH_GREEN_STATE).json()
    assert body["applied_action"] == 0
    assert body["gated_by_min_green"] is False


# --- routing and validation ----------------------------------------------


def test_unknown_intersection_is_a_client_error(client):
    r = client.post("/predict", json={**SWITCHABLE_STATE, "intersection_id": "Z9"})
    assert r.status_code == 404


def test_malformed_payload_is_rejected_at_the_edge(client):
    r = client.post("/predict", json={**SWITCHABLE_STATE, "occupancy": 5.0})
    assert r.status_code == 422


def test_missing_field_is_rejected(client):
    bad = {k: v for k, v in SWITCHABLE_STATE.items() if k != "queue_length"}
    assert client.post("/predict", json=bad).status_code == 422


# --- degradation ----------------------------------------------------------


def test_inference_failure_degrades_to_hold(client):
    policy_service.set_policies(
        {a: BrokenPolicy() for a in settings.agents}, version="broken"
    )
    body = client.post("/predict", json=SWITCHABLE_STATE).json()
    assert body["action_source"] == "fallback"
    assert body["applied_action"] == 0


def test_invalid_action_degrades_to_hold(client):
    policy_service.set_policies(
        {a: InvalidActionPolicy() for a in settings.agents}, version="invalid"
    )
    body = client.post("/predict", json=SWITCHABLE_STATE).json()
    assert body["action_source"] == "fallback"
    assert body["applied_action"] == 0


# --- observability --------------------------------------------------------


def test_metrics_expose_model_and_infra_signals(client):
    client.post("/predict", json=SWITCHABLE_STATE)
    client.post("/predict", json=FRESH_GREEN_STATE)
    text = client.get("/metrics").text
    assert "tsc_predictions_total" in text
    assert "tsc_gated_by_min_green_total" in text
    assert "tsc_input_feature" in text
    assert "tsc_model_ready" in text
