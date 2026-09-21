"""Configuration, sourced entirely from the environment.

Nothing here is hardcoded to a machine or an environment. This is what makes
the same container image promotable from staging to production unchanged:
the image is fixed, the environment supplies the difference.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TSC_", env_file=".env", extra="ignore")

    # --- service ---
    app_name: str = "traffic-signal-policy"
    log_level: str = "INFO"

    # --- model source ---
    # Directory holding one checkpoint per agent, named dqn_<AGENT>_<scenario>.zip
    model_dir: str = "models/grid2x2/dqn/medium"
    scenario: str = "medium"
    # Agents this deployment is expected to serve. Readiness requires ALL of them.
    expected_agents: str = "B1,B2,C1,C2"

    # Set these to resolve models from the MLflow registry instead (phase 2).
    mlflow_tracking_uri: str | None = None
    mlflow_model_prefix: str | None = None  # e.g. "tsc-dqn-2x2-medium"
    # Alias, not stage: MLflow deprecated Staging/Production stages in 2.9.
    mlflow_model_alias: str = "Production"

    # Simulates slow weight loading, used to demonstrate why liveness and
    # readiness probes must be configured separately.
    model_load_delay_seconds: float = 0.0

    # --- control constraints (mirrors traffic_env_v2.py) ---
    # The environment ignores a switch request before this much green has elapsed.
    # The model emits a REQUEST; this gate decides what is safe to apply.
    min_green_seconds: float = 10.0
    decision_interval_seconds: float = 5.0

    # --- observation normalisation (must match training exactly) ---
    # These divisors come from TrafficSignalEnv._get_observation. Changing any
    # of them silently invalidates every trained model.
    norm_queue: float = 20.0
    norm_waiting: float = 200.0
    norm_green_elapsed: float = 60.0
    norm_neighbour_queue: float = 10.0

    @property
    def agents(self) -> list[str]:
        return [a.strip() for a in self.expected_agents.split(",") if a.strip()]


settings = Settings()
