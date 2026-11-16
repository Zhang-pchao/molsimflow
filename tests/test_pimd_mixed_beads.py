import numpy as np
import pytest

from molsimflow.postprocess.pimd.io import prepare_inputs
from molsimflow.postprocess.pimd.mixed_beads import (
    MixedBeadTopology,
    iter_mixed_pimd_path_frames,
)


def expanded(topology):
    values = np.zeros((topology.partitions, len(topology.atom_ids), 3))
    for atom, count in enumerate(topology.bead_counts):
        for b in range(topology.partitions):
            bead = b * count // topology.partitions
            values[b, atom] = [atom + 0.1 * bead, 0.2 * atom, -0.3 * bead]
    return values


def test_nested_topology_unique_weights_and_endpoints():
    topology = MixedBeadTopology((3, 9, 11), (8, 2, 1), 8)
    assert topology.is_mixed and topology.unique_coordinate_count == 11
    np.testing.assert_array_equal(topology.bead_indices()[:, 1], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(topology.representatives().sum(axis=0), [8, 2, 1])
    unique = topology.unique_coordinates(expanded(topology))
    for i, (coordinates, weights) in enumerate(zip(unique, topology.unique_bead_weights())):
        assert weights.sum() == 1
        np.testing.assert_allclose((coordinates * weights[:, None]).sum(axis=0),
                                   expanded(topology)[:, i].mean(axis=0))
    for p in [1, 4]:
        endpoint = MixedBeadTopology((1, 2), (p, p), p)
        assert not endpoint.is_mixed
        assert endpoint.representatives().all()


@pytest.mark.parametrize("ids,counts,p", [
    ((), (), 1), ((1, 1), (1, 1), 1), ((0,), (1,), 1),
    ((1,), (True,), 1), ((1,), (1.0,), 1), ((True,), (1,), 1),
    ((1,), (0,), 1), ((1,), (3,), 4), ((1,), (8,), 4),
    ((1,), (2,), 4), ((1, 2), (1,), 1), ((1,), (1,), True),
])
def test_invalid_quantization_fails(ids, counts, p):
    with pytest.raises(ValueError):
        MixedBeadTopology(ids, counts, p)


def test_metadata_roundtrip_is_independent_of_bias_and_contraction():
    topology = MixedBeadTopology([9, 3], [4, 1], 4)
    assert MixedBeadTopology.from_metadata(topology.to_metadata()) == topology
    assert topology.to_metadata()["nuclear_ensemble"] == "fixed_mixed_bead"
    for key, value in [("lambda", 0.5), ("bias_mode", "centroid_coord")]:
        metadata = topology.to_metadata()
        metadata[key] = value
        with pytest.raises(ValueError, match="schema"):
            MixedBeadTopology.from_metadata(metadata)
    for key, value in [("mapping", "all_pairs"), ("nuclear_ensemble", "full_quantum"),
                       ("atom_ids", None), ("bead_counts", "4,1")]:
        metadata = topology.to_metadata()
        metadata[key] = value
        with pytest.raises(ValueError):
            MixedBeadTopology.from_metadata(metadata)


@pytest.mark.parametrize("pattern", [(1, 1, 1), (4, 4, 4), (8, 2, 1)])
def test_force_pullback_matches_nonlinear_full_configuration_gradient(pattern):
    topology = MixedBeadTopology((1, 2, 3), pattern, max(pattern))
    q = expanded(topology)
    x, y, z = q[:, 0, 0], q[:, 1, 0], q[:, 2, 0]
    forces = np.zeros_like(q)
    forces[:, 0, 0] = -x - 0.2*x*y
    forces[:, 1, 0] = -0.4*y - 0.1*x*x - 0.3*z*z
    forces[:, 2, 0] = -0.6*y*z
    pulled = topology.physical_force_pullback(forces)

    def energy(values):
        a, b, c = values[:, 0, 0], values[:, 1, 0], values[:, 2, 0]
        return np.mean(0.5*a*a + 0.2*b*b + 0.1*a*a*b + 0.3*b*c*c)

    for atom, count in enumerate(pattern):
        stride = topology.partitions // count
        for bead in range(count):
            plus, minus = q.copy(), q.copy()
            region = slice(bead*stride, (bead+1)*stride)
            plus[region, atom, 0] += 1e-6
            minus[region, atom, 0] -= 1e-6
            assert pulled[atom][bead, 0] == pytest.approx(
                -(energy(plus)-energy(minus))/2e-6, abs=1e-8
            )


def test_invalid_shapes_nonfinite_values_and_copy_drift_fail():
    topology = MixedBeadTopology((1, 2), (4, 1), 4)
    q = expanded(topology)
    assert topology.validate_coordinates(q) == 0
    bad = q.copy()
    bad[3, 1, 0] += 1e-5
    with pytest.raises(ValueError, match="disagree"):
        topology.validate_coordinates(bad)
    assert topology.validate_coordinates(bad, tolerance=1e-4) > 0
    for values in [q[0], np.full_like(q, np.nan)]:
        with pytest.raises(ValueError):
            topology.validate_coordinates(values)
        with pytest.raises(ValueError):
            topology.physical_force_pullback(values)
    with pytest.raises(ValueError):
        topology.validate_coordinates(q, tolerance=np.nan)


def write_dumps(tmp_path, topology, *, velocity=True, steps=(0, 4), columns=None):
    labels = columns or ("f_pi[1]", "f_pi[2]", "f_pi[3]")
    fields = ["id", "type", "xu", "yu", "zu", *labels]
    if velocity:
        fields += ["vx", "vy", "vz"]
    q = expanded(topology)
    q[:, -1, 0] -= 12  # Unwrapped paths may lie outside the primary cell.
    indices, owners = topology.bead_indices(), topology.representatives()
    paths = {}
    for b in range(topology.partitions):
        text = ""
        for step in steps:
            text += (f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(topology.atom_ids)}\n"
                     "ITEM: BOX BOUNDS pp pp ff\n0 10\n0 10\n-10 10\n"
                     "ITEM: ATOMS " + " ".join(fields) + "\n")
            order = range(len(topology.atom_ids)) if b % 2 else reversed(range(len(topology.atom_ids)))
            for i in order:
                row = [topology.atom_ids[i], i+1, *q[b, i], topology.bead_counts[i],
                       indices[b, i]+1, int(owners[b, i])]
                if velocity:
                    row += [0.1, 0.2, 0.3]
                text += " ".join(str(value) for value in row) + "\n"
        paths[b] = tmp_path / f"partition-{b}.dump"
        paths[b].write_text(text)
    return paths


def read(paths, topology, **kwargs):
    return list(iter_mixed_pimd_path_frames(
        paths, topology=topology,
        expected_identity={tag: i+1 for i, tag in enumerate(topology.atom_ids)}, **kwargs
    ))


@pytest.mark.parametrize("velocity", [False, True])
def test_native_dump_identity_unwrapped_copies_and_metadata(tmp_path, velocity):
    topology = MixedBeadTopology((9, 3, 17), (8, 2, 1), 8)
    frames = read(write_dumps(tmp_path, topology, velocity=velocity), topology)
    assert [frame.step for frame in frames] == [0, 4]
    assert frames[0].topology == topology
    assert frames[0].maximum_copy_error == 0
    assert (frames[0].velocities is not None) == velocity
    np.testing.assert_array_equal(frames[0].atom_types, [1, 2, 3])
    assert frames[0].positions[0, -1, 0] == -10
    assert len(frames) == 2  # Eight expanded partitions still give two frames.


def test_custom_fix_column_names(tmp_path):
    topology = MixedBeadTopology((1, 2), (2, 1), 2)
    columns = ("f_quantum[1]", "f_quantum[2]", "f_quantum[3]")
    frames = read(write_dumps(tmp_path, topology, columns=columns), topology,
                  metadata_columns=columns)
    assert len(frames) == 2


@pytest.mark.parametrize("mutation,message", [
    ("metadata", "metadata"), ("copy", "disagree"), ("velocity", "disagree"),
    ("wrapped", "xu yu zu"), ("partial_velocity", "vx vy vz"),
    ("cell", "cell must match"), ("duplicate_step", "strictly increasing"),
    ("identity", "ID/type"), ("tail", "format|ended|count"),
])
def test_native_dump_invalid_data_fail_closed(tmp_path, mutation, message):
    topology = MixedBeadTopology((1, 2), (2, 1), 2)
    paths = write_dumps(tmp_path, topology)
    path = paths[1]
    text = path.read_text()
    lines = text.splitlines()
    if mutation in {"metadata", "copy", "velocity", "identity"}:
        header = lines.index(next(line for line in lines if line.startswith("ITEM: ATOMS")))
        fields = lines[header].split()[2:]
        for index in range(header+1, header+3):
            row = lines[index].split()
            if row[0] == "2":
                field, value = {"metadata": ("f_pi[2]", "2"), "copy": ("xu", "-10"),
                                "velocity": ("vx", "0.4"), "identity": ("type", "1")}[mutation]
                row[fields.index(field)] = value
                lines[index] = " ".join(row)
        text = "\n".join(lines)+"\n"
    elif mutation == "wrapped":
        text = text.replace("xu yu zu", "x y z")
    elif mutation == "partial_velocity":
        text = text.replace("vx vy vz", "vx vy unused")
    elif mutation == "cell":
        text = text.replace("0 10\n", "0 11\n", 1)
    elif mutation == "duplicate_step":
        for other in paths.values():
            other.write_text(other.read_text().replace("TIMESTEP\n4", "TIMESTEP\n0"))
        text = path.read_text()
    elif mutation == "tail":
        text += "ITEM: TIMESTEP\n8\nITEM: NUMBER OF ATOMS\n2\n"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read(paths, topology)


def test_missing_partition_duplicate_file_and_different_frame_counts(tmp_path):
    topology = MixedBeadTopology((1, 2), (2, 1), 2)
    paths = write_dumps(tmp_path, topology)
    with pytest.raises(ValueError, match="every zero-based"):
        read({0: paths[0]}, topology)
    with pytest.raises(ValueError, match="distinct"):
        read({0: paths[0], 1: paths[0]}, topology)
    text = paths[1].read_text()
    paths[1].write_text(text[:text.index("ITEM: TIMESTEP\n4")])
    with pytest.raises(ValueError, match="frame counts"):
        read(paths, topology)


def test_legacy_fes_rejects_mixed_ensemble_before_reading_files(tmp_path):
    topology = MixedBeadTopology((1, 2), (4, 1), 4)
    with pytest.raises(ValueError, match="mixed-ensemble output contract"):
        prepare_inputs({"source": {"nuclear_quantization": topology.to_metadata()}},
                       tmp_path, "core")
    uniform = MixedBeadTopology((1, 2), (4, 4), 4)
    with pytest.raises(ValueError, match="partition count"):
        prepare_inputs({"source": {"nuclear_quantization": uniform.to_metadata(),
                                   "bead_colvars": ["one"]}}, tmp_path, "core")

@pytest.mark.parametrize("reserved", [
    "id", "type", "x", "y", "z", "xs", "ys", "zs", "xu", "yu", "zu",
    "xsu", "ysu", "zsu", "ix", "iy", "iz", "vx", "vy", "vz",
])
def test_reader_rejects_reserved_metadata_alias_before_opening_files(tmp_path, reserved):
    topology = MixedBeadTopology((1,), (1,), 1)
    with pytest.raises(ValueError, match="must not alias"):
        list(iter_mixed_pimd_path_frames(
            {0: tmp_path / "missing.dump"}, topology=topology, expected_identity={1: 1},
            metadata_columns=(reserved, "count", "owner"),
        ))


def test_reader_rejects_non_string_metadata_columns(tmp_path):
    topology = MixedBeadTopology((1,), (1,), 1)
    with pytest.raises(ValueError, match="metadata_columns"):
        list(iter_mixed_pimd_path_frames(
            {0: tmp_path / "missing.dump"}, topology=topology, expected_identity={1: 1},
            metadata_columns=([], "count", "owner"),
        ))
