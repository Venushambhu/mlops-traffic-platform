"""Model loading, observation construction and inference.

Two things live here and nowhere else:

1. Where models come from. Swapping local checkpoints for the MLflow registry
   changes this file only.
2. The observation contract. The normalisation below is copied from
   TrafficSignalEnv._get_observation in the thesis environment. If serving
   normalises differently from training, the model receives out-of-distribution
   input and returns confident nonsense — no error, no crash, no alert. Keeping
   it in one place, next to the model, is the whole defence against that.

The deployment is genuinely multi-model: the thesis trains independent learners,
one policy per intersection with no shared parameters, so a request must reach
the model belonging to its own intersection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class Policy(Protocol):
    def predict(self, observation: np.ndarray) -> int: ...


class StableBaselinesPolicy:
    """Wraps one Stable-Baselines3 DQN checkpoint (one intersection's agent)."""

    def __init__(self, model: Any, version: str) -> None:
        self._model = model
        self.version = version

    @classmethod
    def from_path(cls, path: Path) -> "StableBaselinesPolicy":
        from stable_baselines3 import DQN  # lazy: heavy import

        return cls(DQN.load(str(path), device="cpu"), version=f"file:{path.name}")

    def predict(self, observation: np.ndarray) -> int:
        action, _state = self._model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())


class MLflowRegistryPolicy:
    """A model resolved from the MLflow Model Registry by alias.

    The service never names a model version. It asks for whichever version
    currently carries the alias (e.g. "Production"), so promoting a new model
    or rolling back is a registry operation — no config change, no redeploy.

    Aliases, not stages: MLflow deprecated the Staging/Production *stages* in
    2.9 in favour of *aliases*, which are named pointers to a version.
    """

    def __init__(self, model: Any, version: str) -> None:
        self._model = model
        self.version = version

    @classmethod
    def from_registry(cls, name: str, alias: str, tracking_uri: str) -> "MLflowRegistryPolicy":
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")

        try:
            mv = MlflowClient().get_model_version_by_alias(name, alias)
            resolved = f"{name}@{alias}=v{mv.version}"
        except Exception:  # noqa: BLE001 — version detail is useful, not essential
            resolved = f"{name}@{alias}"

        return cls(model, version=resolved)

    def predict(self, observation: np.ndarray) -> int:
        # pyfunc expects a batch; the service predicts one intersection at a time.
        actions = self._model.predict(observation.reshape(1, -1))
        return int(np.asarray(actions).ravel()[0])


class PolicyService:
    """Owns the per-agent model registry and readiness state for this process."""

    def __init__(self) -> None:
        self._policies: dict[str, Policy] = {}
        self._version: str | None = None
        self._load_error: str | None = None

    # --- lifecycle -------------------------------------------------------

    def load(self) -> None:
        """Load every expected agent's model. Called once at startup."""
        if settings.model_load_delay_seconds:
            logger.info("simulating slow load (%.1fs)", settings.model_load_delay_seconds)
            time.sleep(settings.model_load_delay_seconds)

        loaded: dict[str, Policy] = {}
        try:
            for agent in settings.agents:
                if settings.mlflow_model_prefix and settings.mlflow_tracking_uri:
                    name = f"{settings.mlflow_model_prefix}-{agent}"
                    loaded[agent] = MLflowRegistryPolicy.from_registry(
                        name, settings.mlflow_model_alias, settings.mlflow_tracking_uri
                    )
                else:
                    path = Path(settings.model_dir) / f"dqn_{agent}_{settings.scenario}.zip"
                    loaded[agent] = StableBaselinesPolicy.from_path(path)
                logger.info("loaded agent %s", agent)
        except Exception as exc:  # noqa: BLE001 — surfaced via /ready, not swallowed
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.error("model load failed: %s", self._load_error)
            self._policies = {}
            return

        self._policies = loaded
        if settings.mlflow_model_prefix:
            # Report resolved versions, not just the alias, so logs and responses
            # identify the exact artifacts serving traffic. "Production" is a
            # pointer that moves; a version number does not.
            resolved = sorted({getattr(pol, "version", "?") for pol in loaded.values()})
            self._version = ",".join(resolved)
        else:
            parts = Path(settings.model_dir).parts
            # models/grid2x2/dqn/medium -> grid2x2-dqn-medium
            self._version = "-".join(parts[-3:]) if len(parts) >= 3 else settings.scenario
        self._load_error = None
        logger.info("all %d agents ready (version=%s)", len(loaded), self._version)

    def set_policies(self, policies: dict[str, Policy], version: str) -> None:
        """Inject policies directly. Used by tests to avoid loading real weights."""
        self._policies = dict(policies)
        self._version = version
        self._load_error = None

    # --- state -----------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Ready only when EVERY expected agent is loaded.

        A partial load is not 'mostly working' — requests for the missing
        intersection would fail, so the pod should not receive traffic at all.
        """
        return bool(self._policies) and all(a in self._policies for a in settings.agents)

    @property
    def loaded_agents(self) -> list[str]:
        return sorted(self._policies)

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def has_agent(self, agent: str) -> bool:
        return agent in self._policies

    # --- observation -----------------------------------------------------

    @staticmethod
    def build_observation(
        queue_length: float,
        waiting_time: float,
        occupancy: float,
        current_phase: int,
        num_phases: int,
        green_elapsed: float,
        neighbour_queue_total: float,
        num_neighbours: int,
    ) -> np.ndarray:
        """Build the normalised 6-feature vector the trained policy expects.

        Mirrors TrafficSignalEnv._get_observation exactly:
          [0] queue / 20          [3] phase / (num_phases - 1)
          [1] waiting / 200       [4] green_elapsed / 60
          [2] occupancy (0-1)     [5] (neighbour_queue / n_neighbours) / 10

        Every feature is clipped to 1.0. The training space is Box(0, 1, (6,)),
        so an unclipped value is out-of-distribution input.
        """
        return np.array(
            [
                min(queue_length / settings.norm_queue, 1.0),
                min(waiting_time / settings.norm_waiting, 1.0),
                min(occupancy, 1.0),
                min(current_phase / max(num_phases - 1, 1), 1.0),
                min(green_elapsed / settings.norm_green_elapsed, 1.0),
                min(
                    (neighbour_queue_total / max(num_neighbours, 1))
                    / settings.norm_neighbour_queue,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

    # --- inference -------------------------------------------------------

    def predict(self, agent: str, observation: np.ndarray) -> tuple[int, str]:
        """Return (requested_action, action_source).

        An inference failure degrades to hold rather than raising: a signal
        controller that loses its optimiser should keep the current phase, which
        is always legal, instead of losing control entirely.
        """
        policy = self._policies.get(agent)
        if policy is None:
            raise KeyError(agent)

        try:
            action = policy.predict(observation)
        except Exception:  # noqa: BLE001
            logger.exception("inference failed for %s; holding", agent)
            return 0, "fallback"

        if action not in (0, 1):
            logger.warning("policy returned invalid action %s; holding", action)
            return 0, "fallback"

        return action, "policy"

    @staticmethod
    def apply_control_constraints(
        requested_action: int, green_elapsed: float
    ) -> tuple[int, bool]:
        """Gate the policy's request against minimum green time.

        Mirrors the MIN_GREEN check in the training environment. The policy was
        trained against an env that silently ignored early switch requests, so
        enforcing it here keeps serving behaviour consistent with training — and
        stops an unsafe instruction crossing the service boundary at all.

        Returns (applied_action, was_gated).
        """
        if requested_action == 1 and green_elapsed < settings.min_green_seconds:
            return 0, True
        return requested_action, False


policy_service = PolicyService()
