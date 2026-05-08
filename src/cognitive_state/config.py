"""Configuration placeholder for the cognitive state MVP scaffold."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Minimal project-level paths used before runtime configuration exists."""

    project_name: str = "Real-time Cognitive State Estimation from Video"
    artifacts_dir: Path = Path("artifacts")


DEFAULT_CONFIG = ProjectConfig()
