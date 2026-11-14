"""Task runtime primitives for outdoor person following."""

from .motion import OcclusionEvent, RouteFollower, scheduled_visibility
from .runtime import (
    EpisodeResult,
    EpisodeState,
    FailureReason,
    StepObservation,
    TaskEvaluator,
    TerminationThresholds,
)
from .scenario import DisturbanceEvent, ScenarioPlan, build_scenario_plan
from .terrain_routes import (
    GeneratedTerrainRoute, Obstacle2D, TerrainRouteGenerator, terrain_height,
    terrain_mesh_height,
)
from .target_motion import TargetMotionPose, TerrainTargetFollower

__all__ = [
    "EpisodeResult",
    "EpisodeState",
    "DisturbanceEvent",
    "FailureReason",
    "OcclusionEvent",
    "RouteFollower",
    "ScenarioPlan",
    "StepObservation",
    "TaskEvaluator",
    "TerminationThresholds",
    "build_scenario_plan",
    "scheduled_visibility",
    "GeneratedTerrainRoute",
    "Obstacle2D",
    "TerrainRouteGenerator",
    "terrain_height",
    "terrain_mesh_height",
    "TargetMotionPose",
    "TerrainTargetFollower",
]
