"""Command-line interface for molsimflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from molsimflow.io.extxyz import add_pbc_lattice_to_xyz
from molsimflow.io.lammps_data import convert_extxyz_to_lammps_atomic_data
from molsimflow.plumed.double_bubble import generate_double_bubble_plumed
from molsimflow.structure.bubble_geometry import equal_volume_radius
from molsimflow.structure.double_bubble_slab import (
    DoubleBubbleSlabConfig,
    PrebuiltInterfaceConfig,
    SlabBuildConfig,
    build_prebuilt_double_bubble_inputs,
    build_tio2_double_bubble_inputs,
    resolve_molecule_templates,
)
from molsimflow.structure.packmol import run_packmol


def _cmd_add_extxyz_pbc(args: argparse.Namespace) -> int:
    output = add_pbc_lattice_to_xyz(
        xyz_path=args.xyz,
        poscar_path=args.poscar,
        output_path=args.output,
        z_min_padding=args.z_min_padding,
    )
    print(output)
    return 0


def _cmd_extxyz_to_lammps_data(args: argparse.Namespace) -> int:
    output = convert_extxyz_to_lammps_atomic_data(args.xyz, args.output)
    print(output)
    return 0


def _cmd_equal_volume_radius(args: argparse.Namespace) -> int:
    radius = equal_volume_radius(args.radii)
    print(f"{radius:.6f}")
    return 0


def _cmd_config_summary(args: argparse.Namespace) -> int:
    from molsimflow.config.workflow import main as config_main

    return config_main(["summary", "--config", str(args.config)])


def _cmd_config_env(args: argparse.Namespace) -> int:
    from molsimflow.config.workflow import main as config_main

    workflow_args = ["env", "--config", str(args.config)]
    for section in args.section or []:
        workflow_args.extend(["--section", section])
    return config_main(workflow_args)


def _cmd_config_resolve_path(args: argparse.Namespace) -> int:
    from molsimflow.config.workflow import main as config_main

    workflow_args = [
        "resolve-path",
        "--config",
        str(args.config),
        "--section",
        args.section,
        "--key",
        args.key,
    ]
    if args.must_exist:
        workflow_args.append("--must-exist")
    return config_main(workflow_args)


def _print_build_result(result) -> None:
    plan = result.plan
    print(result.output_dir)
    print(f"packmol={result.packmol_path}")
    print(f"slab_xyz={result.slab_xyz_path}")
    print(f"output_xyz={plan.output_xyz_name}")
    print(
        "counts="
        f"h2o:{plan.total_water_count} "
        f"n2:{plan.total_n2_count} "
        f"ion_pairs:{plan.ion_pair_count}"
    )


def _maybe_run_packmol(result, args: argparse.Namespace) -> None:
    if not args.run_packmol:
        return
    run_result = run_packmol(
        result.packmol_path,
        command=args.packmol_command,
        cwd=result.output_dir,
        log_path=args.packmol_log,
    )
    print(f"packmol_run_returncode={run_result.returncode}")
    print(f"packmol_log={run_result.log_path}")
    print(f"packed_xyz={run_result.output_xyz}")


def _cmd_tio2_double_bubble(args: argparse.Namespace) -> int:
    templates = resolve_molecule_templates(
        molecule_dir=args.molecule_dir,
        water_xyz=args.water_xyz,
        n2_xyz=args.n2_xyz,
        cation_xyz=args.cation_xyz,
        anion_xyz=args.anion_xyz,
    )
    slab_config = SlabBuildConfig(
        bulk_structure=args.bulk_structure,
        miller_index=tuple(args.miller_index),
        layers=args.layers,
        vacuum=args.vacuum,
        repeat=tuple(args.repeat),
        bottom_layer_index=args.bottom_layer_index,
        slab_slice_margin=args.slab_slice_margin,
        layer_tolerance=args.layer_tolerance,
        slab_name=args.slab_name,
    )
    target_ph = None if args.no_ph else args.target_ph
    bubble_config = DoubleBubbleSlabConfig(
        gas_radii=tuple(args.gas_radii),
        bubble_spacing=args.bubble_spacing,
        bubble_shape=args.bubble_shape,
        add_interlayer_water=not args.no_interlayer,
        target_ph=target_ph,
        buffer=args.buffer,
        interlayer_thickness=args.interlayer_thickness,
        water_height_radius=args.water_height_radius,
        bubble_center_z_radius=args.bubble_center_z_radius,
        output_system_suffix=args.output_system_suffix,
        packmol_tolerance=args.packmol_tolerance,
        packmol_seed=args.packmol_seed,
    )
    result = build_tio2_double_bubble_inputs(
        slab_config=slab_config,
        bubble_config=bubble_config,
        templates=templates,
        output_dir=args.output_dir,
    )
    _print_build_result(result)
    _maybe_run_packmol(result, args)
    return 0


def _cmd_slab_double_bubble(args: argparse.Namespace) -> int:
    templates = resolve_molecule_templates(
        molecule_dir=args.molecule_dir,
        water_xyz=args.water_xyz,
        n2_xyz=args.n2_xyz,
        cation_xyz=args.cation_xyz,
        anion_xyz=args.anion_xyz,
    )
    target_ph = None if args.no_ph else args.target_ph
    bubble_config = DoubleBubbleSlabConfig(
        gas_radii=tuple(args.gas_radii),
        bubble_spacing=args.bubble_spacing,
        bubble_shape=args.bubble_shape,
        add_interlayer_water=not args.no_interlayer,
        target_ph=target_ph,
        buffer=args.buffer,
        interlayer_thickness=args.interlayer_thickness,
        water_height_radius=args.water_height_radius,
        bubble_center_z_radius=args.bubble_center_z_radius,
        output_system_suffix=args.output_system_suffix,
        packmol_tolerance=args.packmol_tolerance,
        packmol_seed=args.packmol_seed,
    )
    interface_config = PrebuiltInterfaceConfig(
        interface_structure=args.interface_structure,
        layer_tolerance=args.layer_tolerance,
        bottom_layer_index=args.bottom_layer_index,
        bottom_reference_z=args.bottom_reference_z,
        top_reference_z=args.top_reference_z,
        slab_name=args.slab_name,
    )
    result = build_prebuilt_double_bubble_inputs(
        interface_config=interface_config,
        bubble_config=bubble_config,
        templates=templates,
        output_dir=args.output_dir,
    )
    _print_build_result(result)
    _maybe_run_packmol(result, args)
    return 0


def _cmd_plumed_double_bubble(args: argparse.Namespace) -> int:
    summary = generate_double_bubble_plumed(
        data_file=args.data,
        packmol_file=args.packmol,
        output_file=args.output,
        build_py=args.build_py,
        case_label=args.case_label,
    )
    print(args.output)
    print(
        "bubble_radii="
        f"{summary.gas_radius_a:.3f},{summary.gas_radius_b:.3f} "
        f"n2_pairs={len(summary.bubble_a_pairs)},{len(summary.bubble_b_pairs)}"
    )
    return 0


def _plot_common_args(args: argparse.Namespace, kind: str) -> List[str]:
    workflow_args = [
        kind,
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--dpi",
        str(args.dpi),
    ]
    if args.title:
        workflow_args.extend(["--title", args.title])
    if args.width is not None:
        workflow_args.extend(["--width", str(args.width)])
    if args.height is not None:
        workflow_args.extend(["--height", str(args.height)])
    for output_format in args.formats or []:
        workflow_args.extend(["--format", output_format])
    return workflow_args


def _cmd_plot_line(args: argparse.Namespace) -> int:
    from molsimflow.plotting.table_plots import main as plot_main

    workflow_args = _plot_common_args(args, "line")
    workflow_args.extend(["--x-column", args.x_column, "--y-column", args.y_column])
    if args.group_column is not None:
        workflow_args.extend(["--group-column", args.group_column])
    if args.x_label:
        workflow_args.extend(["--x-label", args.x_label])
    if args.y_label:
        workflow_args.extend(["--y-label", args.y_label])
    return plot_main(workflow_args)


def _cmd_plot_scatter(args: argparse.Namespace) -> int:
    from molsimflow.plotting.table_plots import main as plot_main

    workflow_args = _plot_common_args(args, "scatter")
    workflow_args.extend(["--x-column", args.x_column, "--y-column", args.y_column])
    if args.group_column is not None:
        workflow_args.extend(["--group-column", args.group_column])
    if args.label_column is not None:
        workflow_args.extend(["--label-column", args.label_column])
    if args.fit_line:
        workflow_args.append("--fit-line")
    if args.x_label:
        workflow_args.extend(["--x-label", args.x_label])
    if args.y_label:
        workflow_args.extend(["--y-label", args.y_label])
    return plot_main(workflow_args)


def _cmd_plot_heatmap(args: argparse.Namespace) -> int:
    from molsimflow.plotting.table_plots import main as plot_main

    workflow_args = _plot_common_args(args, "heatmap")
    workflow_args.extend(
        [
            "--row-column",
            args.row_column,
            "--column-column",
            args.column_column,
            "--value-column",
            args.value_column,
            "--cmap",
            args.cmap,
        ]
    )
    if args.colorbar_label:
        workflow_args.extend(["--colorbar-label", args.colorbar_label])
    if args.vmin is not None:
        workflow_args.extend(["--vmin", str(args.vmin)])
    if args.vmax is not None:
        workflow_args.extend(["--vmax", str(args.vmax)])
    return plot_main(workflow_args)


def _cmd_postprocess_centroids(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.centroids import main as centroids_main

    workflow_args = [
        "--traj_file",
        str(args.traj_file),
        "--output",
        str(args.output),
        "--cutoff",
        str(args.cutoff),
        "--atom_style",
        args.atom_style,
        "--step_interval",
        str(args.step_interval),
        "--start_frame",
        str(args.start_frame),
        "--end_frame",
        str(args.end_frame),
        "--ions_output",
        str(args.ions_output),
    ]
    if args.data is not None:
        workflow_args.extend(["--data", str(args.data)])
    for flag in [
        "h3o_file",
        "bulk_oh_file",
        "surface_oh_file",
        "surface_h_file",
        "na_file",
        "cl_file",
    ]:
        value = getattr(args, flag)
        if value is not None:
            workflow_args.extend([f"--{flag}", str(value)])
    if args.disable_ions:
        workflow_args.append("--disable_ions")

    return centroids_main(workflow_args)


def _cmd_postprocess_bubble_surface_distance(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.bubble_surface_distance import main as surface_distance_main

    workflow_args = [
        "--traj_file",
        str(args.traj_file),
        "--output",
        str(args.output),
        "--cutoff",
        str(args.cutoff),
        "--atom_style",
        args.atom_style,
        "--step_interval",
        str(args.step_interval),
        "--start_frame",
        str(args.start_frame),
        "--end_frame",
        str(args.end_frame),
        "--surface_fraction",
        str(args.surface_fraction),
        "--min_cluster_size",
        str(args.min_cluster_size),
        "--fs_per_step",
        str(args.fs_per_step),
        "--colvar_match_tolerance_steps",
        str(args.colvar_match_tolerance_steps),
    ]
    for flag in ["data", "max_frames", "nitrogen_type", "colvar_file", "colvar_output"]:
        value = getattr(args, flag)
        if value is not None:
            workflow_args.extend([f"--{flag}", str(value)])
    if args.disable_plot:
        workflow_args.append("--disable_plot")

    return surface_distance_main(workflow_args)


def _cmd_postprocess_coalescence_state(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.coalescence_state import main as coalescence_main

    workflow_args = [
        "--colvar",
        str(args.colvar),
        "--output-dir",
        str(args.output_dir),
        "--start-ns",
        str(args.start_ns),
        "--sample-interval-ns",
        str(args.sample_interval_ns),
        "--time-tolerance-ns",
        str(args.time_tolerance_ns),
        "--bubble-time-tolerance-ns",
        str(args.bubble_time_tolerance_ns),
        "--colvar-time-unit",
        args.colvar_time_unit,
        "--nominal-radius-A",
        str(args.nominal_radius_A),
        "--close-gap-A",
        str(args.close_gap_A),
        "--separated-min-single-fraction",
        str(args.separated_min_single_fraction),
        "--merged-major-total-fraction",
        str(args.merged_major_total_fraction),
        "--merged-minor-total-fraction",
        str(args.merged_minor_total_fraction),
        "--min-persist-samples",
        str(args.min_persist_samples),
        "--cv-bins",
        str(args.cv_bins),
    ]
    if args.colvar_post is not None:
        workflow_args.extend(["--colvar-post", str(args.colvar_post)])
    if args.bubble_evolution is not None:
        workflow_args.extend(["--bubble-evolution", str(args.bubble_evolution)])
    if args.end_ns is not None:
        workflow_args.extend(["--end-ns", str(args.end_ns)])
    if args.rebase_colvar_time_zero:
        workflow_args.append("--rebase-colvar-time-zero")
    if args.surface_contact_distance_A is not None:
        workflow_args.extend(["--surface-contact-distance-A", str(args.surface_contact_distance_A)])
    return coalescence_main(workflow_args)


def _cmd_postprocess_ion_species(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.ion_species import main as ion_species_main

    workflow_args = [
        "--traj",
        str(args.traj),
        "--output-dir",
        str(args.output_dir),
        "--step-interval",
        str(args.step_interval),
        "--start-frame",
        str(args.start_frame),
        "--end-frame",
        str(args.end_frame),
        "--ti-o-cutoff",
        str(args.ti_o_cutoff),
        "--oh-cutoff",
        str(args.oh_cutoff),
        "--max-oh-distance",
        str(args.max_oh_distance),
        "--surface-ti-z-tolerance",
        str(args.surface_ti_z_tolerance),
    ]
    if args.data is not None:
        workflow_args.extend(["--data", str(args.data)])
    if args.atom_style is not None:
        workflow_args.extend(["--atom-style", args.atom_style])
    if args.type_map:
        workflow_args.append("--type-map")
        workflow_args.extend(args.type_map)

    return ion_species_main(workflow_args)


def _cmd_postprocess_ion_z_distribution(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.ion_distribution import main as ion_distribution_main

    workflow_args = [
        "--species-statistics",
        str(args.species_statistics),
        "--output-dir",
        str(args.output_dir),
        "--z-min",
        str(args.z_min),
        "--z-bins",
        str(args.z_bins),
        "--z-range",
        str(args.z_range[0]),
        str(args.z_range[1]),
    ]
    for flag in [
        "h3o_file",
        "bulk_oh_file",
        "surface_oh_file",
        "surface_h_file",
        "na_file",
        "cl_file",
    ]:
        value = getattr(args, flag)
        if value is not None:
            workflow_args.extend([f"--{flag.replace('_', '-')}", str(value)])

    return ion_distribution_main(workflow_args)


def _cmd_postprocess_bridge_water_density(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.bridge_descriptors import main as bridge_main

    workflow_args = [
        "water-density",
        "--input",
        str(args.input),
        "--output-dir",
        str(args.output_dir),
        "--case-label",
        args.case_label,
        "--bridge-radius-A",
        str(args.bridge_radius_A),
        "--bridge-length-A",
        str(args.bridge_length_A),
        "--time-column",
        args.time_column,
        "--gap-column",
        args.gap_column,
        "--water-count-column",
        args.water_count_column,
        "--water-mean-column",
        args.water_mean_column,
        "--gap-bin-width-A",
        str(args.gap_bin_width_A),
        "--min-bin-count",
        str(args.min_bin_count),
    ]
    return bridge_main(workflow_args)


def _cmd_postprocess_bridge_water_dewetting(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.bridge_water_dewetting import main as dewetting_main

    workflow_args = [
        "--dump",
        str(args.dump),
        "--output-dir",
        str(args.output_dir),
        "--water-oxygen-atoms",
        args.water_oxygen_atoms,
        "--colvar-time-unit",
        args.colvar_time_unit,
        "--axis",
        args.axis,
        "--radius-A",
        str(args.radius_A),
        "--lower-A",
        str(args.lower_A),
        "--upper-A",
        str(args.upper_A),
        "--oo-cutoff-A",
        str(args.oo_cutoff_A),
        "--connect-side-thickness-A",
        str(args.connect_side_thickness_A),
        "--connect-min-water",
        str(args.connect_min_water),
        "--time-tolerance-ns",
        str(args.time_tolerance_ns),
        "--cv-bins",
        str(args.cv_bins),
    ]
    for flag in ["plumed", "colvar", "colvar_post"]:
        value = getattr(args, flag)
        if value is not None:
            workflow_args.extend([f"--{flag.replace('_', '-')}", str(value)])
    if args.bubble_a_atoms is not None:
        workflow_args.extend(["--bubble-a-atoms", args.bubble_a_atoms])
    if args.bubble_b_atoms is not None:
        workflow_args.extend(["--bubble-b-atoms", args.bubble_b_atoms])
    if args.dump_time_scale_ns is not None:
        workflow_args.extend(["--dump-time-scale-ns", str(args.dump_time_scale_ns)])
    if args.bulk_number_density_per_A3 is not None:
        workflow_args.extend(["--bulk-number-density-per-A3", str(args.bulk_number_density_per_A3)])
    if args.max_frames is not None:
        workflow_args.extend(["--max-frames", str(args.max_frames)])
    return dewetting_main(workflow_args)


def _bridge_water_dynamics_args(args: argparse.Namespace, command: str) -> List[str]:
    workflow_args = [
        command,
        "--output-dir",
        str(args.output_dir),
        "--gap-source",
        args.gap_source,
        "--gap-bin-width-A",
        str(args.gap_bin_width_A),
        "--min-bin-count",
        str(args.min_bin_count),
        "--state-time-tolerance-ns",
        str(args.state_time_tolerance_ns),
    ]
    if args.manifest is not None:
        workflow_args.extend(["--manifest", str(args.manifest)])
    if args.trace_metrics is not None:
        workflow_args.extend(["--trace-metrics", str(args.trace_metrics)])
    if args.case_label:
        workflow_args.extend(["--case-label", args.case_label])
    if args.state_table is not None:
        workflow_args.extend(["--state-table", str(args.state_table)])
    if args.start_time_ns is not None:
        workflow_args.extend(["--start-time-ns", str(args.start_time_ns)])
    if args.end_time_ns is not None:
        workflow_args.extend(["--end-time-ns", str(args.end_time_ns)])
    return workflow_args


def _cmd_postprocess_bridge_water_flux(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.bridge_water_dynamics import main as dynamics_main

    return dynamics_main(_bridge_water_dynamics_args(args, "flux"))


def _cmd_postprocess_bridge_seed_survival(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.bridge_water_dynamics import main as dynamics_main

    return dynamics_main(_bridge_water_dynamics_args(args, "seed-survival"))


def _cmd_postprocess_bridge_ion_occupancy(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.bridge_descriptors import main as bridge_main

    workflow_args = [
        "ion-occupancy",
        "--positions",
        str(args.positions),
        "--output-dir",
        str(args.output_dir),
        "--case-label",
        args.case_label,
        "--bridge-radius-A",
        str(args.bridge_radius_A),
        "--bridge-length-A",
        str(args.bridge_length_A),
        "--time-column",
        args.time_column,
        "--in-bridge-column",
        args.in_bridge_column,
        "--gap-time-column",
        args.gap_time_column,
        "--gap-column",
        args.gap_column,
        "--gap-state-column",
        args.gap_state_column,
        "--gap-bin-width-A",
        str(args.gap_bin_width_A),
        "--min-bin-count",
        str(args.min_bin_count),
        "--time-tolerance-ns",
        str(args.time_tolerance_ns),
    ]
    if args.gap_table is not None:
        workflow_args.extend(["--gap-table", str(args.gap_table)])
    if args.species_column is not None:
        workflow_args.extend(["--species-column", args.species_column])
    return bridge_main(workflow_args)


def _cmd_postprocess_fes_barriers(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.fes_analysis import main as fes_main

    workflow_args = ["--output-dir", str(args.output_dir)]
    if args.manifest is not None:
        workflow_args.extend(["--manifest", str(args.manifest)])
    for curve in args.curve or []:
        workflow_args.append("--curve")
        workflow_args.extend([str(curve[0]), curve[1], curve[2]])
    for window in args.barrier_window or []:
        workflow_args.extend(["--barrier-window", window])
    workflow_args.extend(
        [
            f"--reference-low={args.reference_low}",
            f"--reference-high={args.reference_high}",
            f"--zero-low={args.zero_low}",
            f"--zero-high={args.zero_high}",
            "--smooth-window",
            str(args.smooth_window),
            "--smooth-passes",
            str(args.smooth_passes),
            "--cv-column",
            str(args.cv_column),
            "--free-energy-column",
            str(args.free_energy_column),
            "--uncertainty-column",
            str(args.uncertainty_column),
        ]
    )
    return fes_main(workflow_args)


def _cmd_postprocess_case_scorecard(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.case_comparison import main as case_main

    workflow_args = [
        "--cases",
        str(args.cases),
        "--case-column",
        args.case_column,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.descriptor_manifest is not None:
        workflow_args.extend(["--descriptor-manifest", str(args.descriptor_manifest)])
    for table in args.descriptor_table or []:
        workflow_args.append("--descriptor-table")
        workflow_args.extend([str(table[0]), str(table[1]), str(table[2]), str(table[3])])
    for pair in args.pair or []:
        workflow_args.extend(["--pair", pair])
    if args.target_column is not None:
        workflow_args.extend(["--target-column", args.target_column])
    if args.correlate is not None:
        workflow_args.extend(["--correlate", args.correlate])
    return case_main(workflow_args)


def _add_centroid_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--traj_file", type=Path, required=True, help="LAMMPS trajectory file path")
    parser.add_argument("--data", type=Path, help="Optional LAMMPS topology/data file path")
    parser.add_argument("--output", default="bubble_centroids.txt", help="Output centroid table file")
    parser.add_argument("--cutoff", type=float, default=5.5, help="N-N clustering cutoff in angstrom")
    parser.add_argument("--atom_style", default="id type x y z", help="LAMMPS atom_style for trajectory parsing")
    parser.add_argument("--step_interval", type=int, default=1, help="Analyze every Nth frame")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index, inclusive")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index, exclusive; -1 means to the end")
    parser.add_argument("--h3o_file", type=Path, help="H3O coordinates file")
    parser.add_argument("--bulk_oh_file", type=Path, help="Bulk OH coordinates file")
    parser.add_argument("--surface_oh_file", type=Path, help="Surface OH coordinates file")
    parser.add_argument("--surface_h_file", type=Path, help="Surface H coordinates file")
    parser.add_argument("--na_file", type=Path, help="Na+ coordinates file")
    parser.add_argument("--cl_file", type=Path, help="Cl- coordinates file")
    parser.add_argument("--ions_output", default="ions_analysis", help="Directory for ion-distance analysis outputs")
    parser.add_argument("--disable_ions", action="store_true", help="Disable ion-distance analysis")


def _add_surface_distance_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--traj_file", type=Path, required=True, help="LAMMPS trajectory file path")
    parser.add_argument("--data", type=Path, help="Optional LAMMPS topology/data file path")
    parser.add_argument("--output", default="bubble_surface_distance.txt", help="Output table path")
    parser.add_argument("--cutoff", type=float, default=5.5, help="N-N clustering cutoff in angstrom")
    parser.add_argument("--atom_style", default="id type x y z", help="LAMMPS atom_style")
    parser.add_argument("--step_interval", type=int, default=1, help="Analyze every Nth frame")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index, inclusive")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index, exclusive; -1 means to the end")
    parser.add_argument("--max_frames", type=int, help="Optional cap on analyzed frames")
    parser.add_argument(
        "--surface_fraction",
        type=float,
        default=0.8,
        help="Surface cutoff as fraction of cluster max radius",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=1,
        help="Minimum atoms required for each of the two bubble clusters",
    )
    parser.add_argument("--nitrogen_type", type=int, help="Optional override for nitrogen atom type")
    parser.add_argument("--fs_per_step", type=float, default=1.0, help="Physical timestep size in fs")
    parser.add_argument("--colvar_file", type=Path, help="Optional COLVAR file for generating COLVAR_surf_dis")
    parser.add_argument("--colvar_output", type=Path, help="Output path for COLVAR_surf_dis")
    parser.add_argument(
        "--colvar_match_tolerance_steps",
        type=int,
        default=0,
        help="Allowed step mismatch when matching trajectory frame to COLVAR row",
    )
    parser.add_argument("--disable_plot", action="store_true", help="Disable time-series plot generation")


def _add_coalescence_state_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--colvar", type=Path, required=True, help="PLUMED COLVAR path with a time column")
    parser.add_argument("--colvar-post", type=Path, help="Optional secondary COLVAR table with cluster counters")
    parser.add_argument("--bubble-evolution", type=Path, help="Optional bubble evolution table")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-ns", type=float, default=0.0)
    parser.add_argument("--end-ns", type=float)
    parser.add_argument("--sample-interval-ns", type=float, default=0.001, help="Use 0 to keep every COLVAR row")
    parser.add_argument("--time-tolerance-ns", type=float, default=0.00051)
    parser.add_argument("--bubble-time-tolerance-ns", type=float, default=0.00051)
    parser.add_argument("--colvar-time-unit", choices=["fs", "ps", "ns"], default="ps")
    parser.add_argument("--rebase-colvar-time-zero", action="store_true")
    parser.add_argument("--nominal-radius-A", type=float, default=19.0)
    parser.add_argument("--surface-contact-distance-A", type=float)
    parser.add_argument("--close-gap-A", type=float, default=0.0)
    parser.add_argument("--separated-min-single-fraction", type=float, default=0.60)
    parser.add_argument("--merged-major-total-fraction", type=float, default=0.85)
    parser.add_argument("--merged-minor-total-fraction", type=float, default=0.10)
    parser.add_argument("--min-persist-samples", type=int, default=3)
    parser.add_argument("--cv-bins", type=int, default=40)


def _add_ion_species_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--traj", type=Path, required=True, help="LAMMPS trajectory file")
    parser.add_argument("--data", type=Path, help="Optional LAMMPS topology/data file")
    parser.add_argument("--output-dir", type=Path, default=Path("ion_analysis_results"))
    parser.add_argument("--atom-style", dest="atom_style", help="Optional MDAnalysis LAMMPS atom_style")
    parser.add_argument("--step-interval", type=int, default=100)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--ti-o-cutoff", type=float, default=3.5)
    parser.add_argument("--oh-cutoff", type=float, default=1.35)
    parser.add_argument("--max-oh-distance", type=float, default=1.8)
    parser.add_argument("--surface-ti-z-tolerance", type=float, default=2.0)
    parser.add_argument("--type-map", nargs="*", help="Atom type mapping entries such as 1=H 2=O 6=Ti")


def _add_ion_z_distribution_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--species-statistics", type=Path, required=True, help="species_statistics.txt path")
    parser.add_argument("--output-dir", type=Path, default=Path("ion_z_distribution_results"))
    parser.add_argument("--h3o-file", type=Path, help="solution_bulk_h3o.xyz")
    parser.add_argument("--bulk-oh-file", type=Path, help="solution_bulk_oh.xyz")
    parser.add_argument("--surface-oh-file", type=Path, help="solution_surface_oh.xyz")
    parser.add_argument("--surface-h-file", type=Path, help="tio2_surface_h.xyz")
    parser.add_argument("--na-file", type=Path, help="na_ions.xyz")
    parser.add_argument("--cl-file", type=Path, help="cl_ions.xyz")
    parser.add_argument("--z-min", type=float, default=15.0, help="Absolute z cutoff before surface subtraction")
    parser.add_argument("--z-bins", type=int, default=100)
    parser.add_argument("--z-range", type=float, nargs=2, default=[0.0, 30.0])


def _add_bridge_water_density_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Input state/metrics CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-label", default="")
    parser.add_argument("--bridge-radius-A", type=float, default=8.0)
    parser.add_argument("--bridge-length-A", type=float, default=20.0)
    parser.add_argument("--time-column", default="time_ns")
    parser.add_argument("--gap-column", default="surface_gap_estimate_A")
    parser.add_argument("--water-count-column", default="bridge_cyl_env.sum")
    parser.add_argument("--water-mean-column", default="bridge_cyl_env.mean")
    parser.add_argument("--gap-bin-width-A", type=float, default=2.0)
    parser.add_argument("--min-bin-count", type=int, default=1)


def _add_bridge_water_dewetting_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dump", type=Path, required=True, help="LAMMPS dump trajectory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--water-oxygen-atoms", required=True, help="PLUMED-style atom id expression for water oxygens")
    parser.add_argument("--plumed", type=Path, help="PLUMED file containing bubA_all and bubB_all labels")
    parser.add_argument("--bubble-a-atoms", help="Explicit atom expression for bubble A")
    parser.add_argument("--bubble-b-atoms", help="Explicit atom expression for bubble B")
    parser.add_argument("--colvar", type=Path, help="Optional COLVAR table")
    parser.add_argument("--colvar-post", type=Path, help="Optional secondary COLVAR table")
    parser.add_argument("--colvar-time-unit", choices=["fs", "ps", "ns"], default="ns")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--radius-A", type=float, default=6.5)
    parser.add_argument("--lower-A", type=float, default=-8.0)
    parser.add_argument("--upper-A", type=float, default=8.0)
    parser.add_argument("--oo-cutoff-A", type=float, default=3.5)
    parser.add_argument("--connect-side-thickness-A", type=float, default=2.0)
    parser.add_argument("--connect-min-water", type=int, default=2)
    parser.add_argument("--time-tolerance-ns", type=float, default=0.00051)
    parser.add_argument("--dump-time-scale-ns", type=float)
    parser.add_argument("--bulk-number-density-per-A3", type=float)
    parser.add_argument("--cv-bins", type=int, default=40)
    parser.add_argument("--max-frames", type=int)


def _add_bridge_water_dynamics_postprocess_args(parser: argparse.ArgumentParser) -> None:
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path, help="CSV with case_label,trace_metrics,state_table columns")
    input_group.add_argument("--trace-metrics", type=Path, help="Single bridge_water_trace_metrics.csv input")
    parser.add_argument("--case-label", default="", help="Case label for --trace-metrics")
    parser.add_argument("--state-table", type=Path, help="Optional coalescence_state_table.csv for --trace-metrics")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-time-ns", type=float)
    parser.add_argument("--end-time-ns", type=float)
    parser.add_argument("--gap-source", choices=["trace", "coalescence"], default="coalescence")
    parser.add_argument("--gap-bin-width-A", type=float, default=2.0)
    parser.add_argument("--min-bin-count", type=int, default=1)
    parser.add_argument("--state-time-tolerance-ns", type=float, default=0.0015)


def _add_bridge_ion_occupancy_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--positions", type=Path, required=True, help="tracked_bridge_ion_positions.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-label", default="")
    parser.add_argument("--gap-table", type=Path, help="Optional state table with surface gap")
    parser.add_argument("--bridge-radius-A", type=float, default=8.0)
    parser.add_argument("--bridge-length-A", type=float, default=20.0)
    parser.add_argument("--time-column", default="time_ns")
    parser.add_argument("--species-column")
    parser.add_argument("--in-bridge-column", default="in_bridge_region")
    parser.add_argument("--gap-time-column", default="time_ns")
    parser.add_argument("--gap-column", default="surface_gap_estimate_A")
    parser.add_argument("--gap-state-column", default="state")
    parser.add_argument("--gap-bin-width-A", type=float, default=2.0)
    parser.add_argument("--min-bin-count", type=int, default=1)
    parser.add_argument("--time-tolerance-ns", type=float, default=0.0015)


def _add_fes_barriers_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, help="CSV with path,label,group,dataset_key columns")
    parser.add_argument(
        "--curve",
        nargs=3,
        action="append",
        metavar=("PATH", "LABEL", "GROUP"),
        help="Explicit curve input; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--barrier-window", action="append", help="Window as name:low:high; may be repeated")
    parser.add_argument("--reference-low", type=float, default=float("-inf"))
    parser.add_argument("--reference-high", type=float, default=float("inf"))
    parser.add_argument("--zero-low", type=float, default=float("-inf"))
    parser.add_argument("--zero-high", type=float, default=float("inf"))
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--smooth-passes", type=int, default=1)
    parser.add_argument("--cv-column", type=int, default=0, help="Zero-based CV column index")
    parser.add_argument("--free-energy-column", type=int, default=1, help="Zero-based free-energy column index")
    parser.add_argument("--uncertainty-column", type=int, default=2, help="Zero-based uncertainty column index; use -1 to disable")


def _add_plot_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Input CSV table")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path or stem")
    parser.add_argument("--format", dest="formats", action="append", help="Output format; may be repeated")
    parser.add_argument("--title", default="")
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("--dpi", type=int, default=300)


def _add_case_scorecard_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cases", type=Path, required=True, help="Case manifest CSV with case_label or selected case column")
    parser.add_argument("--case-column", default="case_label", help="Case key column in --cases")
    parser.add_argument("--descriptor-manifest", type=Path, help="CSV with name,path,case_column,columns columns")
    parser.add_argument(
        "--descriptor-table",
        nargs=4,
        action="append",
        metavar=("NAME", "PATH", "CASE_COLUMN", "COLUMNS"),
        help="Explicit descriptor table; COLUMNS may be comma, semicolon, or pipe separated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", action="append", help="Case pair as REFERENCE:TARGET[:LABEL]; may be repeated")
    parser.add_argument("--target-column", help="Scorecard column used as the correlation target")
    parser.add_argument("--correlate", help="Comma, semicolon, or pipe separated scorecard columns for deltas/correlations")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="molsimflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config = subparsers.add_parser("config", help="Workflow configuration helpers")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)

    config_summary = config_subparsers.add_parser("summary", help="Print config sections and keys")
    config_summary.add_argument("--config", type=Path, required=True)
    config_summary.set_defaults(func=_cmd_config_summary)

    config_env = config_subparsers.add_parser("env", help="Print shell export lines for selected config sections")
    config_env.add_argument("--config", type=Path, required=True)
    config_env.add_argument("--section", action="append", help="Section to export; may be repeated")
    config_env.set_defaults(func=_cmd_config_env)

    config_resolve = config_subparsers.add_parser("resolve-path", help="Resolve one path value relative to config")
    config_resolve.add_argument("--config", type=Path, required=True)
    config_resolve.add_argument("--section", required=True)
    config_resolve.add_argument("--key", required=True)
    config_resolve.add_argument("--must-exist", action="store_true")
    config_resolve.set_defaults(func=_cmd_config_resolve_path)

    structure = subparsers.add_parser("structure", help="Structure preparation utilities")
    structure_subparsers = structure.add_subparsers(dest="structure_command", required=True)

    add_pbc = structure_subparsers.add_parser(
        "add-extxyz-pbc",
        help="Add extended XYZ PBC/lattice metadata using a POSCAR lattice",
    )
    add_pbc.add_argument("--poscar", type=Path, required=True, help="POSCAR file with lattice vectors")
    add_pbc.add_argument("--xyz", type=Path, required=True, help="Input XYZ file")
    add_pbc.add_argument("--output", type=Path, required=True, help="Output extended XYZ file")
    add_pbc.add_argument(
        "--z-min-padding",
        type=float,
        default=5.0,
        help="Shift coordinates so the minimum z coordinate equals this value",
    )
    add_pbc.set_defaults(func=_cmd_add_extxyz_pbc)

    to_lammps = structure_subparsers.add_parser(
        "extxyz-to-lammps-data",
        help="Convert an orthorhombic extended XYZ file to LAMMPS atom_style atomic data",
    )
    to_lammps.add_argument("--xyz", type=Path, required=True, help="Input extended XYZ file")
    to_lammps.add_argument("--output", type=Path, required=True, help="Output LAMMPS data file")
    to_lammps.set_defaults(func=_cmd_extxyz_to_lammps_data)

    equal_radius = structure_subparsers.add_parser(
        "equal-volume-radius",
        help="Compute a same-size sphere radius preserving mean gas volume",
    )
    equal_radius.add_argument("--radii", type=float, nargs="+", required=True, help="Input radii")
    equal_radius.set_defaults(func=_cmd_equal_volume_radius)

    generic_builder = structure_subparsers.add_parser(
        "slab-double-bubble",
        help="Build double-bubble PACKMOL inputs from a prebuilt interface structure",
    )
    generic_builder.add_argument(
        "--interface-structure",
        type=Path,
        required=True,
        help="Prebuilt interface structure file readable by ASE",
    )
    generic_builder.add_argument("--output-dir", type=Path, required=True, help="Directory for generated files")
    generic_builder.add_argument(
        "--molecule-dir",
        type=Path,
        help="Directory containing H2O.xyz, N2.xyz, Na.xyz, and OH-.xyz templates",
    )
    generic_builder.add_argument("--water-xyz", type=Path, help="Explicit water template path")
    generic_builder.add_argument("--n2-xyz", type=Path, help="Explicit N2 template path")
    generic_builder.add_argument("--cation-xyz", type=Path, help="Explicit cation template path")
    generic_builder.add_argument("--anion-xyz", type=Path, help="Explicit anion template path")
    generic_builder.add_argument("--gas-radii", type=float, nargs=2, default=[19.0, 14.0], help="Two gas radii")
    generic_builder.add_argument("--bubble-spacing", type=float, default=50.0, help="Bubble center spacing")
    generic_builder.add_argument("--bubble-shape", choices=["sphere", "cylinder"], default="sphere")
    generic_builder.add_argument("--target-ph", type=float, default=13.4, help="Upper-water target pH")
    generic_builder.add_argument("--no-ph", action="store_true", help="Do not add pH-control ion templates")
    generic_builder.add_argument("--no-interlayer", action="store_true", help="Disable interlayer water")
    generic_builder.add_argument("--buffer", type=float, default=2.2, help="Main water/slab buffer in Angstrom")
    generic_builder.add_argument(
        "--interlayer-thickness", type=float, default=2.0, help="Interlayer water thickness in Angstrom"
    )
    generic_builder.add_argument(
        "--water-height-radius",
        type=float,
        help="Radius reference for upper-water height; defaults to the first gas radius",
    )
    generic_builder.add_argument(
        "--bubble-center-z-radius",
        type=float,
        help="Radius reference for bubble z-center; defaults to the first gas radius",
    )
    generic_builder.add_argument(
        "--bottom-reference-z",
        type=float,
        help="Explicit lower-interface z reference; otherwise inferred from layers",
    )
    generic_builder.add_argument(
        "--top-reference-z",
        type=float,
        help="Explicit upper-interface z reference; otherwise inferred from layers",
    )
    generic_builder.add_argument(
        "--bottom-layer-index",
        type=int,
        default=0,
        help="Layer index used for lower-water reference when --bottom-reference-z is omitted",
    )
    generic_builder.add_argument("--layer-tolerance", type=float, default=0.1)
    generic_builder.add_argument("--slab-name", default="interface_slab")
    generic_builder.add_argument("--output-system-suffix", default="slab")
    generic_builder.add_argument("--packmol-tolerance", type=float, default=2.4)
    generic_builder.add_argument("--packmol-seed", type=int, default=-1)
    generic_builder.add_argument(
        "--run-packmol",
        action="store_true",
        help="Run Packmol after writing packmol.in",
    )
    generic_builder.add_argument(
        "--packmol-command",
        default="packmol",
        help="Packmol executable or command string used with --run-packmol",
    )
    generic_builder.add_argument(
        "--packmol-log",
        type=Path,
        help="Packmol stdout/stderr log path; defaults to packmol.out in the output directory",
    )
    generic_builder.set_defaults(func=_cmd_slab_double_bubble)

    tio2_builder = structure_subparsers.add_parser(
        "tio2-double-bubble",
        help="Build TiO2 double-bubble PACKMOL inputs and interface slab files",
    )
    tio2_builder.add_argument("--bulk-structure", type=Path, required=True, help="Bulk structure file read by ASE")
    tio2_builder.add_argument("--output-dir", type=Path, required=True, help="Directory for generated files")
    tio2_builder.add_argument(
        "--molecule-dir",
        type=Path,
        help="Directory containing H2O.xyz, N2.xyz, Na.xyz, and OH-.xyz templates",
    )
    tio2_builder.add_argument("--water-xyz", type=Path, help="Explicit water template path")
    tio2_builder.add_argument("--n2-xyz", type=Path, help="Explicit N2 template path")
    tio2_builder.add_argument("--cation-xyz", type=Path, help="Explicit cation template path")
    tio2_builder.add_argument("--anion-xyz", type=Path, help="Explicit anion template path")
    tio2_builder.add_argument("--gas-radii", type=float, nargs=2, default=[19.0, 14.0], help="Two gas radii")
    tio2_builder.add_argument("--bubble-spacing", type=float, default=50.0, help="Bubble center spacing")
    tio2_builder.add_argument("--bubble-shape", choices=["sphere", "cylinder"], default="sphere")
    tio2_builder.add_argument("--target-ph", type=float, default=13.4, help="Upper-water target pH")
    tio2_builder.add_argument("--no-ph", action="store_true", help="Do not add pH-control ion templates")
    tio2_builder.add_argument("--no-interlayer", action="store_true", help="Disable interlayer water")
    tio2_builder.add_argument("--buffer", type=float, default=2.2, help="Main water/slab buffer in Angstrom")
    tio2_builder.add_argument(
        "--interlayer-thickness", type=float, default=2.0, help="Interlayer water thickness in Angstrom"
    )
    tio2_builder.add_argument(
        "--water-height-radius",
        type=float,
        help="Radius reference for upper-water height; defaults to the first gas radius",
    )
    tio2_builder.add_argument(
        "--bubble-center-z-radius",
        type=float,
        help="Radius reference for bubble z-center; defaults to the first gas radius",
    )
    tio2_builder.add_argument("--miller-index", type=int, nargs=3, default=[1, 0, 1])
    tio2_builder.add_argument("--layers", type=int, default=6)
    tio2_builder.add_argument("--vacuum", type=float, default=100.0)
    tio2_builder.add_argument("--repeat", type=int, nargs=3, default=[11, 20, 1])
    tio2_builder.add_argument("--bottom-layer-index", type=int, default=4)
    tio2_builder.add_argument("--slab-slice-margin", type=float, default=0.5)
    tio2_builder.add_argument("--layer-tolerance", type=float, default=0.1)
    tio2_builder.add_argument("--slab-name", default="tio2_interface_101")
    tio2_builder.add_argument("--output-system-suffix", default="tio2")
    tio2_builder.add_argument("--packmol-tolerance", type=float, default=2.4)
    tio2_builder.add_argument("--packmol-seed", type=int, default=-1)
    tio2_builder.add_argument(
        "--run-packmol",
        action="store_true",
        help="Run Packmol after writing packmol.in",
    )
    tio2_builder.add_argument(
        "--packmol-command",
        default="packmol",
        help="Packmol executable or command string used with --run-packmol",
    )
    tio2_builder.add_argument(
        "--packmol-log",
        type=Path,
        help="Packmol stdout/stderr log path; defaults to packmol.out in the output directory",
    )
    tio2_builder.set_defaults(func=_cmd_tio2_double_bubble)

    plumed = subparsers.add_parser("plumed", help="PLUMED input generators")
    plumed_subparsers = plumed.add_subparsers(dest="plumed_command", required=True)

    double_bubble = plumed_subparsers.add_parser(
        "double-bubble",
        help="Generate a double-bubble PLUMED file from PACKMOL and LAMMPS data",
    )
    double_bubble.add_argument("--data", type=Path, required=True, help="LAMMPS data file")
    double_bubble.add_argument("--packmol", type=Path, required=True, help="PACKMOL input file")
    double_bubble.add_argument("--output", type=Path, required=True, help="Output PLUMED file")
    double_bubble.add_argument(
        "--build-py",
        type=Path,
        help="Optional structure-build Python file used for radius/spacing fallback metadata",
    )
    double_bubble.add_argument("--case-label", default="", help="Optional case label for PLUMED comments")
    double_bubble.set_defaults(func=_cmd_plumed_double_bubble)

    plot = subparsers.add_parser("plot", help="CSV-driven plotting helpers")
    plot_subparsers = plot.add_subparsers(dest="plot_kind", required=True)

    line = plot_subparsers.add_parser("line", help="Plot line series from a CSV table")
    _add_plot_common_args(line)
    line.add_argument("--x-column", required=True)
    line.add_argument("--y-column", required=True)
    line.add_argument("--group-column")
    line.add_argument("--x-label", default="")
    line.add_argument("--y-label", default="")
    line.set_defaults(func=_cmd_plot_line)

    scatter = plot_subparsers.add_parser("scatter", help="Plot a scatter table")
    _add_plot_common_args(scatter)
    scatter.add_argument("--x-column", required=True)
    scatter.add_argument("--y-column", required=True)
    scatter.add_argument("--group-column")
    scatter.add_argument("--label-column")
    scatter.add_argument("--fit-line", action="store_true")
    scatter.add_argument("--x-label", default="")
    scatter.add_argument("--y-label", default="")
    scatter.set_defaults(func=_cmd_plot_scatter)

    heatmap = plot_subparsers.add_parser("heatmap", help="Plot a heatmap from long-form CSV rows")
    _add_plot_common_args(heatmap)
    heatmap.add_argument("--row-column", required=True)
    heatmap.add_argument("--column-column", required=True)
    heatmap.add_argument("--value-column", required=True)
    heatmap.add_argument("--colorbar-label", default="")
    heatmap.add_argument("--cmap", default="RdBu_r")
    heatmap.add_argument("--vmin", type=float)
    heatmap.add_argument("--vmax", type=float)
    heatmap.set_defaults(func=_cmd_plot_heatmap)

    postprocess = subparsers.add_parser("postprocess", help="MD post-processing workflows")
    postprocess_subparsers = postprocess.add_subparsers(
        dest="postprocess_command",
        required=True,
    )

    centroids = postprocess_subparsers.add_parser(
        "centroids",
        help="Compute two-bubble centroids from a LAMMPS trajectory",
    )
    _add_centroid_postprocess_args(centroids)
    centroids.set_defaults(func=_cmd_postprocess_centroids)

    surface_distance = postprocess_subparsers.add_parser(
        "bubble-surface-distance",
        help="Analyze two-bubble centroid and surface distances",
    )
    _add_surface_distance_postprocess_args(surface_distance)
    surface_distance.set_defaults(func=_cmd_postprocess_bubble_surface_distance)

    coalescence_state = postprocess_subparsers.add_parser(
        "coalescence-state",
        help="Assign provisional two-bubble coalescence states from COLVAR tables",
    )
    _add_coalescence_state_postprocess_args(coalescence_state)
    coalescence_state.set_defaults(func=_cmd_postprocess_coalescence_state)

    ion_species = postprocess_subparsers.add_parser(
        "ion-species",
        help="Classify ion species from a LAMMPS trajectory",
    )
    _add_ion_species_postprocess_args(ion_species)
    ion_species.set_defaults(func=_cmd_postprocess_ion_species)

    ion_z_distribution = postprocess_subparsers.add_parser(
        "ion-z-distribution",
        help="Compute ion z-distributions from classified ion XYZ files",
    )
    _add_ion_z_distribution_postprocess_args(ion_z_distribution)
    ion_z_distribution.set_defaults(func=_cmd_postprocess_ion_z_distribution)

    bridge_water = postprocess_subparsers.add_parser(
        "bridge-water-density",
        help="Compute bridge-water density descriptors from a state/metrics table",
    )
    _add_bridge_water_density_postprocess_args(bridge_water)
    bridge_water.set_defaults(func=_cmd_postprocess_bridge_water_density)

    bridge_dewetting = postprocess_subparsers.add_parser(
        "bridge-water-dewetting",
        help="Compute bridge-water dewetting and connectivity metrics from a LAMMPS dump",
    )
    _add_bridge_water_dewetting_postprocess_args(bridge_dewetting)
    bridge_dewetting.set_defaults(func=_cmd_postprocess_bridge_water_dewetting)

    bridge_flux = postprocess_subparsers.add_parser(
        "bridge-water-flux",
        help="Compute bridge-water entry/exit flux proxy summaries",
    )
    _add_bridge_water_dynamics_postprocess_args(bridge_flux)
    bridge_flux.set_defaults(func=_cmd_postprocess_bridge_water_flux)

    bridge_seed_survival = postprocess_subparsers.add_parser(
        "bridge-seed-survival",
        help="Compute seed bridge-water survival proxy summaries",
    )
    _add_bridge_water_dynamics_postprocess_args(bridge_seed_survival)
    bridge_seed_survival.set_defaults(func=_cmd_postprocess_bridge_seed_survival)

    bridge_ion = postprocess_subparsers.add_parser(
        "bridge-ion-occupancy",
        help="Compute strict bridge-ion occupancy and charge descriptors",
    )
    _add_bridge_ion_occupancy_postprocess_args(bridge_ion)
    bridge_ion.set_defaults(func=_cmd_postprocess_bridge_ion_occupancy)

    fes_barriers = postprocess_subparsers.add_parser(
        "fes-barriers",
        help="Process 1D FES curves and compute barrier summaries",
    )
    _add_fes_barriers_postprocess_args(fes_barriers)
    fes_barriers.set_defaults(func=_cmd_postprocess_fes_barriers)

    case_scorecard = postprocess_subparsers.add_parser(
        "case-scorecard",
        help="Join case descriptor tables and compute case deltas/correlations",
    )
    _add_case_scorecard_postprocess_args(case_scorecard)
    case_scorecard.set_defaults(func=_cmd_postprocess_case_scorecard)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
