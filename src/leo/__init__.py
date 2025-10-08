"""LEO research project utilities."""

from importlib import resources as _resources
from pathlib import Path as _Path


def project_root() -> "_Path":
    """Return the repository root directory."""
    return _Path(__file__).resolve().parents[2]


__all__ = ["project_root"]
