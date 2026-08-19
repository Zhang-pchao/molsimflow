import csv
import json

from molsimflow.postprocess.electronic_path import build_electronic_path_profiles


def _write_tsv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_build_electronic_path_profiles_applies_geometry_gate(tmp_path):
    atom_table = tmp_path / "atoms.tsv"
    _write_tsv(
        atom_table,
        [
            {
                "system": "demo",
                "stratum": stratum,
                "replicate": replicate,
                "atom_index_one_based": 4,
                "element": "C",
                "charge": charge,
                "spin": spin,
            }
            for stratum, replicate, charge, spin in [
                ("n", "1", -0.1, 0.2),
                ("n", "2", 0.1, 0.4),
                ("a", "1", 0.5, 0.6),
                ("a", "2", 0.7, 0.8),
            ]
        ],
    )
    frame_table = tmp_path / "frames.tsv"
    _write_tsv(
        frame_table,
        [
            {
                "system": "demo",
                "stratum": stratum,
                "replicate": replicate,
                "geometry_state": geometry,
                "organic_charge": organic_charge,
                "organic_spin": organic_spin,
            }
            for stratum, replicate, geometry, organic_charge, organic_spin in [
                ("n", "1", "ion-like", -0.8, 0.9),
                ("n", "2", "ion-like", -0.7, 1.0),
                ("a", "1", "product-like", 0.2, 0.8),
                ("a", "2", "product-like", 0.4, 1.0),
            ]
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "systems": {
                    "demo": {
                        "atom_labels": {"C_d": 4},
                        "routes": {
                            "path": {
                                "states": [
                                    {
                                        "label": "N",
                                        "stratum": "n",
                                        "expected_geometry_state": "reactant-like",
                                    },
                                    {
                                        "label": "Z",
                                        "stratum": "a",
                                        "expected_geometry_state": "product-like",
                                    },
                                ]
                            }
                        },
                    }
                }
            }
        )
    )

    outputs = build_electronic_path_profiles(
        atom_table,
        frame_table,
        config,
        tmp_path / "outputs",
        make_plots=False,
    )

    with outputs["summary"].open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = {
        (row["state_label"], row["descriptor_label"]): row
        for row in rows
    }
    assert selected[("N", "C_d")]["n_total"] == "2"
    assert selected[("N", "C_d")]["n_admitted"] == "0"
    assert selected[("N", "C_d")]["mean_charge_all"] == "0.0"
    assert selected[("N", "C_d")]["mean_charge_admitted"] == ""
    assert selected[("Z", "C_d")]["n_admitted"] == "2"
    assert selected[("Z", "C_d")]["mean_charge_admitted"] == "0.6"
    assert selected[("Z", "organic_total")]["mean_spin_admitted"] == "0.9"
