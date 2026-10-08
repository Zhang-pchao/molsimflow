import numpy as np

from molsimflow.postprocess.dpa4c_training_tsne import (
    energy_per_atom,
    family_frame_indices,
    choose_label_offset,
    FAMILY_TITLES,
    formula_key,
    load_deepmd_npy,
    l2_normalize_descriptors,
    read_table_s4,
    sample_indices,
    select_unique_compositions,
)
from molsimflow.postprocess.deepmd_dataset_sketch import DatasetBundle


def test_sample_indices_are_uniform():
    assert sample_indices(10, 3).tolist() == [0, 4, 9]


def test_load_deepmd_npy_reads_uniform_frames_and_counts_all(tmp_path):
    dataset = tmp_path / "system"
    set_dir = dataset / "set.000"
    set_dir.mkdir(parents=True)
    (dataset / "type_map.raw").write_text("H O\n")
    (dataset / "type.raw").write_text("0\n1\n")
    np.save(set_dir / "coord.npy", np.arange(18, dtype=float).reshape(3, 6))
    np.save(set_dir / "box.npy", np.tile(np.eye(3).reshape(1, 9), (3, 1)))
    np.save(set_dir / "energy.npy", np.array([-2.0, -1.5, -1.0]))

    coords, cells, energies, atom_types, type_symbols, available, selected = load_deepmd_npy(
        dataset, 2
    )

    assert coords.shape == (2, 2, 3)
    assert cells.shape == (2, 3, 3)
    assert energies.tolist() == [-2.0, -1.0]
    assert atom_types == [0, 1]
    assert type_symbols == ["H", "O"]
    assert available == 3
    assert selected.tolist() == [0, 2]


def test_energy_per_atom_uses_reference_energy():
    bundle = DatasetBundle(
        coords_list=[],
        cells_list=[],
        energies=np.array([-4.0, -9.0]),
        n_atoms_list=[2, 3],
        atom_types_list=[],
        type_symbols_list=[],
        system_configs=[],
        source_paths=[],
        frame_to_dataset=[],
    )

    assert energy_per_atom(bundle).tolist() == [-2.0, -3.0]


def test_l2_normalize_descriptors_matches_dpeva_structure_normalization():
    normalized = l2_normalize_descriptors(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.0, 0.0])


def test_family_frame_indices_follow_dataset_offsets():
    grouped = family_frame_indices(
        [{"family": "hon_primary"}, {"family": "sio2_dft"}], [0, 2, 5]
    )
    assert grouped["hon_primary"].tolist() == [0, 1]
    assert grouped["sio2_dft"].tolist() == [2, 3, 4]


def test_choose_label_offset_avoids_an_occupied_box():
    first = choose_label_offset((100.0, 100.0), "1", [], (0.0, 0.0, 200.0, 200.0), 72.0)
    second = choose_label_offset(
        (100.0, 100.0), "2", [first[2]], (0.0, 0.0, 200.0, 200.0), 72.0
    )
    assert first[:2] != second[:2]


def test_family_titles_cover_all_training_blocks():
    assert set(FAMILY_TITLES) == {
        "hon_primary", "hon_rest", "nacl_95pct", "tio2_95pct", "sio2_dft"
    }


def test_select_unique_compositions_prefers_non_distilled(tmp_path):
    rows = []
    for index, (block, symbols, types) in enumerate([
        ("hon_primary", "H O\n", "0\n1\n"),
        ("hon_rest", "H O\n", "0\n1\n"),
        ("sio2_dft", "H O Si\n", "0\n1\n2\n"),
        ("sio2_distilled", "H O Si\n", "0\n1\n2\n"),
    ]):
        system = tmp_path / str(index)
        system.mkdir()
        (system / "type_map.raw").write_text(symbols)
        (system / "type.raw").write_text(types)
        rows.append({"index": str(index), "block": block, "system_path": str(system)})

    selected = select_unique_compositions(
        rows,
        ordered_keys=[formula_key("HOSi"), formula_key("HO")],
        expected_count=2,
    )

    assert [row["block"] for row in selected] == ["sio2_dft", "hon_primary"]
    assert [row["original_manifest_index"] for row in selected] == ["2", "0"]


def test_read_table_s4_reuses_ids_for_repeated_compositions(tmp_path):
    table = tmp_path / "si.tex"
    table.write_text(
        "\\label{tab:dpa4c_inventory}\n"
        "\\num{1} & \\ce{H2O1} & SCAN DFT & 1 \\\\\n"
        "\\ce{H1O1} & SCAN DFT & 1 \\\\\n"
        "\\num{1} & \\ce{H2O1} & DPA4 pseudo-label & 1 \\\\\n"
        "\\end{longtable}\n"
    )

    order, rows = read_table_s4(table, expected_rows=3, expected_systems=2)

    assert len(order) == 2
    assert [row["subsystem_id"] for row in rows] == [1, 2, 1]
