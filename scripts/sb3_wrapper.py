"""MLflow wrapper for a Stable-Baselines3 DQN checkpoint.

MLflow's registry works with models it knows how to load. SB3 is not one of its
built-in flavours, so we wrap the checkpoint in a `pyfunc` model: MLflow stores
the .zip as an artifact, and this class tells it how to load and call that file.

The payoff is that a served model can be addressed as
`models:/<name>/Production` instead of a file path — the service asks for
"whatever is in production" rather than naming a specific file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import mlflow.models
import mlflow.pyfunc


class SB3DQNWrapper(mlflow.pyfunc.PythonModel):
    """Loads an SB3 DQN checkpoint and exposes deterministic action selection."""

    def load_context(self, context) -> None:
        """Called once by MLflow when the model is loaded, not per prediction."""
        from stable_baselines3 import DQN

        self.model = DQN.load(context.artifacts["checkpoint"], device="cpu")

    def predict(self, context, model_input, params=None):
        """Return one action per input row.

        Accepts a DataFrame or array of shape (n, 6) — the normalised
        observation vectors. Returns an int array of actions (0 = hold,
        1 = switch).
        """
        if isinstance(model_input, pd.DataFrame):
            observations = model_input.to_numpy(dtype=np.float32)
        else:
            observations = np.asarray(model_input, dtype=np.float32)

        if observations.ndim == 1:
            observations = observations.reshape(1, -1)

        actions, _states = self.model.predict(observations, deterministic=True)
        return np.asarray(actions).astype(int).ravel()


# MLflow "models from code": rather than pickling an instance of this class,
# MLflow stores THIS FILE alongside the model and re-executes it at load time.
# Pickling the object instead fails with ModuleNotFoundError, because the
# pickle refers to a module that does not exist in the loading process.
mlflow.models.set_model(SB3DQNWrapper())
