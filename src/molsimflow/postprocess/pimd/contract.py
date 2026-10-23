"""Legacy contract adapters; field-specific scientific meanings stay separate."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

ANALYSIS_PROFILES = {"core", "water_ionization_opes"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def analysis_profile(contract: Mapping[str, object]) -> str:
    """Return the generic or case-specific diagnostics profile."""
    profile = str(contract.get("analysis_profile", "water_ionization_opes"))
    require(profile in ANALYSIS_PROFILES, f"unsupported analysis profile: {profile}")
    return profile


def artifact_basename(config: Mapping[str, object], key: str, default: str) -> str:
    """Return a configured output basename without allowing path traversal."""
    name = str(config.get(key, default))
    require(bool(name) and Path(name).name == name, f"{key} must be a basename")
    return name


def cv_column_names(
    config: Mapping[str, object], key: str, logical_names: Sequence[str]
) -> Tuple[str, ...]:
    """Resolve representation-specific PLUMED columns for logical CV names."""
    names = tuple(str(value) for value in config.get(key, logical_names))
    require(len(names) == len(logical_names), f"{key} must match cv_names")
    return names


def diagnostic_cv_spec(config: Mapping[str, object] | None) -> Dict[str, object] | None:
    """Validate one directly printed diagnostic CV without defining a FES."""
    if config is None:
        return None
    require(isinstance(config, Mapping), "diagnostic_cv must be an object")
    result = {
        "name": str(config.get("name", "")),
        "sampling_column": str(config.get("sampling_column", "")),
        "bead_column": str(config.get("bead_column", "")),
        "label": str(config.get("label", config.get("name", ""))),
        "mean_tolerance": float(config.get("mean_tolerance", 1e-12)),
    }
    for key in ("name", "sampling_column", "bead_column"):
        value = str(result[key])
        require(bool(value) and Path(value).name == value, f"invalid diagnostic_cv {key}")
    require(float(result["mean_tolerance"]) >= 0.0, "negative diagnostic_cv tolerance")
    return result
