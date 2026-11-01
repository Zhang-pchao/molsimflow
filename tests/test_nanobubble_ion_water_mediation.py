"""Synthetic tests for fixed-grid and defect-carrier separation."""

import csv
import gzip
import pytest

from molsimflow.postprocess.nanobubble_ion_water_mediation import (
    minimum_image,
    renewal,
    run_case,
)


def test_minimum_image_and_sampled_renewal():
    assert minimum_image(9.0, 10.0) == -1.0
    turnover, retained, net = renewal({1, 2}, {2, 3})
    assert turnover == pytest.approx(2 / 3)
    assert retained == 0.5
    assert net == 0.0


def _dump_frame(step, points):
    lines = [
        "ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(len(points)),
        "ITEM: BOX BOUNDS pp pp ff", "0 10", "0 10", "0 20",
        "ITEM: ATOMS id type x y z",
    ]
    lines.extend(f"{atom_id} 2 {x} {y} 1" for atom_id, x, y in points)
    return "\n".join(lines) + "\n"


def test_fixed_cell_defect_occupancy_outlives_carrier(tmp_path):
    state = tmp_path / "state.lammpstrj"
    state.write_text(
        _dump_frame(0, [(1, 1, 1), (2, 2, 1), (3, 8, 8), (4, 8, 7)])
        + _dump_frame(20, [(1, 8, 6), (2, 2, 1), (3, 1, 1), (4, 8, 7)])
    )
    geometry = tmp_path / "geometry.tsv"
    with geometry.open("w") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=[
            "step", "bubble_center_x_A", "bubble_center_y_A",
            "advancing_position_A", "receding_position_A",
            "transverse_width_q05_q95_A",
        ])
        writer.writeheader()
        for step in (0, 20):
            writer.writerow(dict(step=step, bubble_center_x_A=2.5, bubble_center_y_A=2.5,
                                 advancing_position_A=2.5, receding_position_A=7.5,
                                 transverse_width_q05_q95_A=5))
    ions = tmp_path / "ions.csv.gz"
    with gzip.open(ions, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "species", "atom_id", "x_A", "y_A", "z_A"])
        writer.writeheader()
        writer.writerow(dict(step=0, species="H3O_plus_candidate", atom_id=1, x_A=1, y_A=1, z_A=1))
        writer.writerow(dict(step=20, species="H3O_plus_candidate", atom_id=3, x_A=1, y_A=1, z_A=1))
    output = tmp_path / "output"
    result = run_case({
        "case_id": "synthetic", "direction": "x", "first_step": 0, "last_step": 20,
        "stride_steps": 20, "timestep_fs": 0.5, "geometry_tsv": str(geometry),
        "species": ["H3O_plus_candidate"], "surface_z_A": 0, "layer_height_A": 8,
        "cell_width_A": 5, "minimum_liquid_oxygen": 1,
        "edge_halfwidth_A": 2, "transverse_padding_A": 0.1,
        "expected_atoms": 4, "solution_atom_range": [1, 4],
        "ion_sources": [{"path": str(ions), "minimum_step": 0, "maximum_step": 20}],
        "state_sources": [{"path": str(state), "minimum_step": 0, "maximum_step": 20}],
    }, output)
    assert result["frames"] == 2
    with (output / "fixed_grid_renewal.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    exposed = [r for r in rows if r["region"] == "front" and r["ion_present_at_start"] == "1"]
    assert len(exposed) == 1
    assert exposed[0]["defect_cell_persistent"] == "1"
    assert exposed[0]["same_carrier_persistent"] == "0"
