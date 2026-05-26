import csv
import math

from molsimflow.postprocess.bridge_water_dewetting import (
    BridgeWaterDewettingConfig,
    analyze_bridge_connectivity,
    analyze_bridge_water_dewetting,
    extract_bubble_atom_groups_from_plumed,
    parse_atom_selection,
)


def _write_dump(path):
    with path.open("w") as handle:
        for timestep, waters in [
            (0, [(10.0, 10.0, 8.0), (10.0, 10.0, 10.0), (10.0, 10.0, 12.0)]),
            (1, [(1.0, 1.0, 1.0), (10.0, 10.0, 10.0), (19.0, 19.0, 19.0)]),
        ]:
            atoms = [
                (1, 1, 10.0, 10.0, 6.0),
                (2, 1, 10.2, 10.0, 6.0),
                (3, 1, 10.0, 10.0, 14.0),
                (4, 1, 10.2, 10.0, 14.0),
            ]
            atoms.extend((atom_id, 2, x, y, z) for atom_id, (x, y, z) in zip([5, 6, 7], waters))
            handle.write("ITEM: TIMESTEP\n")
            handle.write(f"{timestep}\n")
            handle.write("ITEM: NUMBER OF ATOMS\n")
            handle.write(f"{len(atoms)}\n")
            handle.write("ITEM: BOX BOUNDS pp pp pp\n")
            handle.write("0 20\n0 20\n0 20\n")
            handle.write("ITEM: ATOMS id type x y z\n")
            for atom in atoms:
                handle.write("%d %d %.3f %.3f %.3f\n" % atom)


def _write_colvar(path):
    with path.open("w") as handle:
        handle.write("#! FIELDS time d3d_all bridge_cyl_env.sum\n")
        handle.write("0.000 6.0 3\n")
        handle.write("0.001 7.0 1\n")


def test_parse_atom_selection_and_plumed_groups(tmp_path):
    assert parse_atom_selection("1,3-7:2") == [1, 3, 5, 7]

    plumed = tmp_path / "in.plumed"
    plumed.write_text(
        "\n".join(
            [
                "bubA_core: GROUP ATOMS=1-2",
                "bubA_all: GROUP ATOMS=bubA_core",
                "bubB_all: GROUP ATOMS=3,4",
            ]
        ),
        encoding="utf-8",
    )

    bubble_a, bubble_b = extract_bubble_atom_groups_from_plumed(plumed)

    assert bubble_a == [1, 2]
    assert bubble_b == [3, 4]


def test_bridge_connectivity_detects_spanning_cluster():
    import numpy as np

    coords = np.array([[5.0, 5.0, 3.0], [5.0, 5.0, 5.0], [5.0, 5.0, 7.0]])
    axial = np.array([-2.0, 0.0, 2.0])
    bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])

    largest, connected = analyze_bridge_connectivity(
        coords,
        axial,
        bounds,
        oo_cutoff_A=2.1,
        lower_A=-2.0,
        upper_A=2.0,
        side_thickness_A=0.25,
    )

    assert largest == 3
    assert connected is True


def test_analyze_bridge_water_dewetting_writes_outputs(tmp_path):
    dump = tmp_path / "dump.lammpstrj"
    colvar = tmp_path / "COLVAR"
    _write_dump(dump)
    _write_colvar(colvar)

    outputs = analyze_bridge_water_dewetting(
        dump=dump,
        colvar=colvar,
        output_dir=tmp_path / "out",
        water_oxygen_atoms=[5, 6, 7],
        bubble_a_atoms=[1, 2],
        bubble_b_atoms=[3, 4],
        colvar_time_unit="ns",
        config=BridgeWaterDewettingConfig(
            axis="z",
            radius_A=1.5,
            lower_A=-2.0,
            upper_A=2.0,
            oo_cutoff_A=2.1,
            connect_side_thickness_A=0.25,
            dump_time_scale_ns=0.001,
            bulk_number_density_per_A3=0.1,
            cv_bins=2,
        ),
    )

    assert outputs["frame_table"].exists()
    assert outputs["cv_summary"].exists()
    assert outputs["statistics"].exists()

    with outputs["frame_table"].open() as handle:
        rows = list(csv.DictReader(handle))

    assert [int(float(row["Nw_bridge"])) for row in rows] == [3, 1]
    assert rows[0]["water_bridge_connected_flag"] == "True"
    assert math.isclose(float(rows[0]["d3d_all"]), 6.0)
