import csv
import json

import numpy as np

from molsimflow.cli import build_parser
from molsimflow.postprocess.deepmd_dataset_sketch import (
    downsample_by_config,
    find_deepmd_datasets,
    group_configs_by_chemistry,
    select_representative_per_config,
    write_embedding_csv,
    write_subsystem_latex,
)


def test_find_deepmd_datasets_discovers_roots_with_set_dirs(tmp_path):
    dataset = tmp_path / "root" / "system_a"
    (dataset / "set.000").mkdir(parents=True)
    (dataset / "type.raw").write_text("0\n", encoding="utf-8")
    ignored = tmp_path / "root" / "not_dataset"
    ignored.mkdir()

    assert find_deepmd_datasets([tmp_path / "root"]) == [dataset]


def test_downsample_by_config_caps_each_group_uniformly():
    indices = np.arange(8)
    configs = ["A", "A", "A", "A", "A", "B", "B", "B"]

    selected = downsample_by_config(indices, configs, max_per_config=2)

    assert selected.tolist() == [0, 2, 5, 6]


def test_select_representative_per_config_uses_centroid_nearest_point():
    embedding = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ]
    )
    configs = ["A", "A", "B", "B"]

    reps = select_representative_per_config(embedding, configs)

    assert reps == {"A": 0, "B": 2}


def test_group_configs_by_chemistry_groups_by_element_set_then_name():
    mapping = group_configs_by_chemistry(["2H1O", "2H1N", "1C2H1O"])

    assert mapping == {"1C2H1O": 1, "2H1N": 2, "2H1O": 3}


def test_write_embedding_csv(tmp_path):
    output = tmp_path / "points.csv"
    write_embedding_csv(
        output,
        embedding=np.array([[1.5, 2.5]]),
        energies_per_atom=np.array([-8.0]),
        selected_indices=np.array([42]),
        system_configs=["2H1O"],
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "pc1": "1.5",
            "pc2": "2.5",
            "energy_per_atom": "-8.0",
            "original_frame_index": "42",
            "system_config": "2H1O",
        }
    ]


def test_write_embedding_csv_includes_dataset_metadata(tmp_path):
    output = tmp_path / "points.csv"
    write_embedding_csv(
        output,
        embedding=np.array([[1.5, 2.5]]),
        energies_per_atom=np.array([-8.0]),
        selected_indices=np.array([42]),
        system_configs=["2H1O"],
        source_paths=["/data/set_a"],
        dataset_numbers={"/data/set_a": 7},
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "pc1": "1.5",
            "pc2": "2.5",
            "energy_per_atom": "-8.0",
            "original_frame_index": "42",
            "system_config": "2H1O",
            "source_path": "/data/set_a",
            "dataset_number": "7",
        }
    ]


def test_write_subsystem_latex_contains_counts(tmp_path):
    output = tmp_path / "subsystem_table.tex"
    write_subsystem_latex(output, {1: "2H1O", 2: "2H1N"}, {"2H1O": 3, "2H1N": 4})

    text = output.read_text(encoding="utf-8")
    assert "total number of frames is 7" in text
    assert "1 & 2H1O & 3" in text
    assert "2 & 2H1N & 4" in text


def test_top_level_parser_exposes_deepmd_dataset_sketch_without_optional_imports(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "postprocess",
            "deepmd-dataset-sketch",
            "--dataset",
            str(tmp_path / "data"),
            "--model",
            str(tmp_path / "frozen_model.pb"),
            "--output",
            str(tmp_path / "out"),
            "--method",
            "pca",
            "--tsne-early-exaggeration",
            "4",
            "--tsne-learning-rate",
            "200",
            "--descriptor-preprocess",
            "standardize-pca",
            "--descriptor-pca-components",
            "50",
        ]
    )

    assert args.postprocess_command == "deepmd-dataset-sketch"
    assert args.method == "pca"
    assert args.tsne_early_exaggeration == 4.0
    assert args.tsne_learning_rate == 200.0
    assert args.descriptor_preprocess == "standardize-pca"
    assert args.descriptor_pca_components == 50
    assert json.dumps({"ok": True})
