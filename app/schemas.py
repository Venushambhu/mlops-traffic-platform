"""Request/response contract for the traffic signal policy service.

The caller sends RAW physical quantities (vehicle counts, seconds). The service
owns the normalisation, because the divisors are part of the model's training
contract — not something every client should have to know or keep in sync.
"""

from typing import List, Literal

from pydantic import BaseModel, Field


class TrafficState(BaseModel):
    """Raw observation for one signalised intersection, as SUMO/TraCI reports it."""

    intersection_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Traffic light system ID — must be one of the served agents.",
        examples=["B1"],
    )
    queue_length: float = Field(
        ...,
        ge=0,
        description="Total halting vehicles across this intersection's controlled lanes.",
        examples=[12],
    )
    waiting_time: float = Field(
        ...,
        ge=0,
        description="Total accumulated waiting time in seconds across controlled lanes.",
        examples=[90.0],
    )
    occupancy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean lane occupancy, already a 0-1 fraction from TraCI.",
        examples=[0.4],
    )
    current_phase: int = Field(
        ..., ge=0, description="Index of the active signal phase."
    )
    num_phases: int = Field(
        ..., ge=1, le=16, description="Total phases in this intersection's programme."
    )
    green_elapsed: float = Field(
        ...,
        ge=0,
        description="Seconds of green already served in the current phase.",
        examples=[25.0],
    )
    neighbour_queue_total: float = Field(
        ...,
        ge=0,
        description="Summed halting vehicles across all neighbouring intersections.",
        examples=[8],
    )
    num_neighbours: int = Field(
        ..., ge=0, description="How many neighbours that total covers."
    )


class PredictionResponse(BaseModel):
    """Policy request plus the action that is actually safe to apply."""

    intersection_id: str

    requested_action: int = Field(
        ..., description="What the policy asked for: 0 = hold, 1 = switch."
    )
    applied_action: int = Field(
        ...,
        description=(
            "What may safely be executed. Equals requested_action unless a "
            "control constraint overrode it."
        ),
    )
    action_label: Literal["hold", "switch"]
    gated_by_min_green: bool = Field(
        ...,
        description=(
            "True when the policy requested a switch before minimum green had "
            "elapsed and the service downgraded it to hold."
        ),
    )

    action_source: Literal["policy", "fallback"] = Field(
        ...,
        description="'fallback' when the policy could not be applied and a safe default was used.",
    )
    model_version: str
    observation: List[float] = Field(
        ..., description="The normalised 6-feature vector actually sent to the model."
    )
    inference_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    ready: bool
    agents_loaded: List[str]
    agents_expected: List[str]
    model_version: str | None = None
    detail: str | None = None
