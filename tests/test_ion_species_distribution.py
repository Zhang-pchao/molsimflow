import math

import numpy as np

from molsimflow.postprocess.ion_distribution import (
    analyze_ion_z_distribution,
    compute_ion_z_distributions,
    load_species_xyz,
)
from molsimflow.postprocess.ion_species import (
    IonSpeciesConfig,
    classify_ion_species,
    write_species_statistics,
    write_species_xyz,
)


def test_classify_ion_species_counts_tio2_and_solution_species():
    symbols = ["O", "Ti", "Ti", "H", "O", "H", "O", "H", "H", "H", "Na", "Cl"]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 1.8],
            [0.0, 0.0, 0.9],
            [0.5, 0.0, 3.2],
            [0.5, 0.0, 4.1],
            [8.0, 8.0, 8.0],
            [8.9, 8.0, 8.0],
            [8.0, 8.9, 8.0],
            [8.0, 8.0, 8.9],
            [2.0, 2.0, 7.0],
            [3.0, 2.0, 7.0],
        ]
    )

    result = classify_ion_species(
        symbols=symbols,
        positions=positions,
        box_dims=np.array([20.0, 20.0, 20.0]),
        frame_index=7,
        config=IonSpeciesConfig(oh_cutoff=1.2),
    )

    assert result.frame_index == 7
    assert result.count("tio2_surface_h") == 1
    assert result.count("solution_surface_oh") == 1
    assert result.count("solution_bulk_h3o") == 1
    assert result.count("na_ions") == 1
    assert result.count("cl_ions") == 1


def test_ion_distribution_reads_species_xyz_and_filters_by_absolute_z(tmp_path):
    result0 = classify_ion_species(
        symbols=["O", "Ti", "Ti", "O", "H", "H", "H", "Na"],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 1.8],
                [8.0, 8.0, 17.0],
                [8.9, 8.0, 17.0],
                [8.0, 8.9, 17.0],
                [8.0, 8.0, 17.9],
                [2.0, 2.0, 20.0],
            ]
        ),
        box_dims=np.array([30.0, 30.0, 30.0]),
        frame_index=0,
    )
    result1 = classify_ion_species(
        symbols=["O", "Ti", "Ti", "O", "H", "H", "H", "Na"],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 1.8],
                [8.0, 8.0, 10.0],
                [8.9, 8.0, 10.0],
                [8.0, 8.9, 10.0],
                [8.0, 8.0, 10.9],
                [2.0, 2.0, 21.0],
            ]
        ),
        box_dims=np.array([30.0, 30.0, 30.0]),
        frame_index=1,
    )

    output_dir = tmp_path / "ion_species"
    write_species_xyz(result0, output_dir=output_dir, box_info=[30.0, 30.0, 30.0], append=False)
    write_species_xyz(result1, output_dir=output_dir, box_info=[30.0, 30.0, 30.0], append=True)
    write_species_statistics([result0, result1], output_dir / "species_statistics.txt")

    species_frames = {
        "h3o": load_species_xyz(output_dir / "solution_bulk_h3o.xyz", "h3o"),
        "na": load_species_xyz(output_dir / "na_ions.xyz", "na"),
    }
    distributions = compute_ion_z_distributions(
        species_frames,
        surface_z_by_frame={0: 2.0, 1: 2.5},
        z_min_threshold=15.0,
        z_bins=3,
        z_range=(0.0, 30.0),
    )

    assert distributions["h3o"].total_count == 1
    assert distributions["h3o"].filtered_count == 1
    assert math.isclose(distributions["h3o"].z_coords[0], 15.0)
    assert distributions["na"].total_count == 2
    assert math.isclose(distributions["na"].avg_per_frame, 1.0)

    dist_dir = tmp_path / "distribution"
    written = analyze_ion_z_distribution(
        species_statistics=output_dir / "species_statistics.txt",
        species_files={
            "h3o": output_dir / "solution_bulk_h3o.xyz",
            "na": output_dir / "na_ions.xyz",
        },
        output_dir=dist_dir,
        z_min_threshold=15.0,
        z_bins=3,
        z_range=(0.0, 30.0),
    )

    assert set(written) == {"h3o", "na"}
    assert (dist_dir / "ion_z_distribution_summary.tsv").exists()
    assert (dist_dir / "ion_z_density.tsv").exists()
