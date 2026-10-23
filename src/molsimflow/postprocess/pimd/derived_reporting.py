"""PIMD analysis orchestration with explicit stage boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from .observables import inverse_piecewise_logdistance, transform_piecewise_logdistance_fes
from .reporting import FES_TABLE_FIELDS, fes_table_columns, sampling_protocol_label, write_csv
from .types import AlignedPaths, JointResults, PreparedInputs, WeightedPaths
from .units import EV_TO_KCAL_MOL


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def write_derived_coordinate(
    contract: Mapping[str, object],
    output: Path,
    profile: str,
    inputs: PreparedInputs,
    aligned: AlignedPaths,
    weighted: WeightedPaths,
    joint: JointResults,
    _plotter,
):
    reweight = inputs.reweight
    bias_mode = inputs.bias_mode
    primary_estimator = inputs.primary_estimator
    cv_names = aligned.cv_names
    sampling_label = aligned.sampling_label
    derived_spec = aligned.derived_spec
    cv_labels = aligned.cv_labels
    beads = aligned.beads
    derived_validation = aligned.derived_validation
    kbt_ev = weighted.kbt_ev
    x_grid = joint.x_grid
    y_grid = joint.y_grid
    primary_name = joint.primary_name
    primary_supports = joint.primary_supports
    primary_kcal = joint.primary_kcal
    marginal_tables = joint.marginal_tables
    derived_coordinate_summary = None
    if derived_spec is not None:
        require(derived_validation is not None, "derived-coordinate validation absent")
        source_name = str(derived_spec["source"])
        target_name = str(derived_spec["target"])
        source_component = cv_names.index(source_name)
        source_grid = x_grid if source_component == 0 else y_grid
        transform_options = {
            "switch": float(derived_spec["switch"]),
            "offset": float(derived_spec["offset"]),
            "linear_shift": float(derived_spec["linear_shift"]),
            "log_scale": float(derived_spec["log_scale"]),
            "log_reference": float(derived_spec["log_reference"]),
        }
        source_marginals = marginal_tables[source_name]
        derived_curves_kcal: Dict[str, np.ndarray] = {}
        target_grid = None
        density_jacobian = None
        for key in ("centroid", "probability_mean", "free_energy_mean"):
            current_grid, current_jacobian, current_curve = transform_piecewise_logdistance_fes(
                source_grid,
                source_marginals[key],
                kbt_ev * EV_TO_KCAL_MOL,
                zero_minimum=False,
                **transform_options,
            )
            if target_grid is None:
                target_grid = current_grid
                density_jacobian = current_jacobian
            else:
                require(np.array_equal(current_grid, target_grid), "derived grids differ")
                require(
                    np.array_equal(current_jacobian, density_jacobian),
                    "derived Jacobians differ",
                )
            derived_curves_kcal[key] = current_curve
        require(target_grid is not None and density_jacobian is not None, "derived grid absent")
        primary_zero = np.min(
            derived_curves_kcal[primary_estimator][source_marginals[f"{primary_estimator}_support"]]
        )
        for key in ("probability_mean", "free_energy_mean"):
            derived_curves_kcal[key] -= primary_zero
        derived_curves_kcal["centroid"] -= np.min(
            derived_curves_kcal["centroid"][source_marginals["centroid_support"]]
        )
        derived_marginal_supports = {
            key: np.asarray(source_marginals[f"{key}_support"], dtype=bool).copy()
            for key in (
                "centroid",
                "probability_mean",
                "free_energy_mean",
                "common",
                "estimator_common",
                "all_common",
                "sampling_probability_common",
                "sampling_free_energy_common",
            )
        }
        require(
            all(
                np.array_equal(
                    derived_marginal_supports[key],
                    source_marginals[f"{key}_support"],
                )
                for key in derived_marginal_supports
            ),
            "derived marginal support changed",
        )
        jacobian_name = f"d{source_name}_d{target_name}"
        write_csv(
            output / "fes1d" / f"{target_name}.csv",
            (
                {
                    target_name: target_grid[index],
                    source_name: source_grid[index],
                    jacobian_name: density_jacobian[index],
                    **fes_table_columns(
                        derived_curves_kcal,
                        derived_marginal_supports,
                        primary_estimator,
                        index,
                    ),
                }
                for index in range(len(target_grid))
            ),
            [
                target_name,
                source_name,
                jacobian_name,
                *FES_TABLE_FIELDS,
            ],
        )
        _plotter.plot_fes1d(
            output,
            target_name,
            target_grid,
            derived_curves_kcal,
            derived_marginal_supports,
            float(reweight["plot_max_kcal_mol"]),
            str(derived_spec["label"]),
            sampling_label=sampling_label,
            bias_mode=bias_mode,
        )

        surface_axis = 1 if source_component == 0 else 0
        derived_surfaces_kcal: Dict[str, np.ndarray] = {}
        for key in ("centroid", "probability_mean", "free_energy_mean"):
            current_grid, current_jacobian, current_surface = transform_piecewise_logdistance_fes(
                source_grid,
                primary_kcal[key],
                kbt_ev * EV_TO_KCAL_MOL,
                axis=surface_axis,
                zero_minimum=False,
                **transform_options,
            )
            require(np.array_equal(current_grid, target_grid), "derived 1D/2D grids differ")
            require(
                np.array_equal(current_jacobian, density_jacobian),
                "derived 1D/2D Jacobians differ",
            )
            derived_surfaces_kcal[key] = current_surface
        primary_zero = np.min(
            derived_surfaces_kcal[primary_estimator][primary_supports[primary_estimator]]
        )
        for key in ("probability_mean", "free_energy_mean"):
            derived_surfaces_kcal[key] -= primary_zero
        derived_surfaces_kcal["centroid"] -= np.min(
            derived_surfaces_kcal["centroid"][primary_supports["centroid"]]
        )
        derived_supports = {
            key: np.asarray(primary_supports[key], dtype=bool).copy()
            for key in (
                "centroid",
                "probability_mean",
                "free_energy_mean",
                "common",
                "estimator_common",
                "all_common",
                "sampling_probability_common",
                "sampling_free_energy_common",
            )
        }
        require(
            all(
                np.array_equal(derived_supports[key], primary_supports[key])
                for key in derived_supports
            ),
            "derived 2D support changed",
        )
        derived_names = list(cv_names)
        derived_names[source_component] = target_name
        derived_labels = list(cv_labels)
        derived_labels[source_component] = str(derived_spec["label"])
        derived_x_grid = target_grid if source_component == 0 else x_grid
        derived_y_grid = y_grid if source_component == 0 else target_grid
        derived_rows = []
        for iy, y_value in enumerate(derived_y_grid):
            for ix, x_value in enumerate(derived_x_grid):
                source_index = ix if source_component == 0 else iy
                derived_rows.append(
                    {
                        derived_names[0]: x_value,
                        derived_names[1]: y_value,
                        source_name: source_grid[source_index],
                        jacobian_name: density_jacobian[source_index],
                        **fes_table_columns(
                            derived_surfaces_kcal,
                            derived_supports,
                            primary_estimator,
                            (iy, ix),
                        ),
                    }
                )
        derived_surface_name = f"{primary_name}-{'-'.join(derived_names)}"
        write_csv(
            output / "fes2d" / f"{derived_surface_name}.csv",
            derived_rows,
            [
                derived_names[0],
                derived_names[1],
                source_name,
                jacobian_name,
                *FES_TABLE_FIELDS,
            ],
        )
        _plotter.plot_fes2d(
            output,
            derived_x_grid,
            derived_y_grid,
            derived_surfaces_kcal,
            derived_supports,
            float(reweight["plot_max_kcal_mol"]),
            derived_labels,
            suffix=f"-{derived_surface_name}",
            sampling_label=sampling_label,
            bias_mode=bias_mode,
            protocol_label=sampling_protocol_label(reweight),
            bead_count=beads.shape[1],
        )
        source_zoom = contract["plots"].get("fes_zoom")
        derived_zoom = None
        if source_zoom is not None:
            derived_zoom = [list(bounds) for bounds in source_zoom]
            derived_zoom[source_component] = list(
                inverse_piecewise_logdistance(source_zoom[source_component], **transform_options)
            )
            _plotter.plot_fes2d(
                output,
                derived_x_grid,
                derived_y_grid,
                derived_surfaces_kcal,
                derived_supports,
                float(reweight["plot_max_kcal_mol"]),
                derived_labels,
                zoom=derived_zoom,
                suffix=f"-{derived_surface_name}-sampled-region",
                sampling_label=sampling_label,
                bias_mode=bias_mode,
                protocol_label=sampling_protocol_label(reweight),
                bead_count=beads.shape[1],
            )
        derived_coordinate_summary = {
            "kind": str(derived_spec["kind"]),
            "source": source_name,
            "target": target_name,
            "printed_column": str(derived_spec["printed_column"]),
            "printed_transform_tolerance": float(derived_spec["printed_transform_tolerance"]),
            "printed_transform_validation": derived_validation,
            "density_jacobian": jacobian_name,
            "sampling_support_unchanged": True,
            "weighting_unchanged": True,
            "independent_cv": False,
            "fes1d": f"fes1d/{target_name}.csv",
            "fes2d": f"fes2d/{derived_surface_name}.csv",
            "figure": f"figures/fes2d-comparison-{derived_surface_name}.png",
            "sampled_region_figure": (
                f"figures/fes2d-comparison-{derived_surface_name}-sampled-region.png"
                if derived_zoom is not None
                else None
            ),
        }
    return derived_coordinate_summary
