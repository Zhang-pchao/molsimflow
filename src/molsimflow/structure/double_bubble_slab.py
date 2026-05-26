"""Configurable double-bubble slab structure preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from molsimflow.structure.bubble_geometry import (
    cylinder_volume,
    molecule_count_from_density,
    sphere_volume,
    two_sphere_intersection_volume,
)
from molsimflow.structure.packmol import (
    PackmolInput,
    PackmolStructure,
    resolve_template_path,
    validate_template_paths,
    write_packmol_input,
)
from molsimflow.structure.regions import BoxRegion, CylinderRegion, SphereRegion


PathLike = Union[str, Path]


@dataclass(frozen=True)
class MoleculeTemplates:
    """Paths to molecule templates used in PACKMOL input generation."""

    water: Path
    nitrogen: Path
    cation: Optional[Path] = None
    anion: Optional[Path] = None


@dataclass(frozen=True)
class DoubleBubbleSlabConfig:
    """Configuration for a slab plus two gas bubbles and surrounding solvent."""

    gas_radii: Tuple[float, float] = (19.0, 14.0)
    bubble_spacing: Optional[float] = 50.0
    bubble_shape: str = "sphere"
    add_interlayer_water: bool = True
    target_ph: Optional[float] = 13.4
    buffer: float = 2.2
    xy_buffer: Optional[float] = None
    z_slab_buffer: Optional[float] = None
    interlayer_thickness: float = 2.0
    z_shift: Optional[float] = None
    water_height_radius: Optional[float] = None
    bubble_center_z_radius: Optional[float] = None
    water_density_kg_m3: float = 950.0
    water_molar_mass_g_mol: float = 18.015
    n2_density_kg_m3: float = 350.0
    n2_molar_mass_g_mol: float = 28.014
    cation_label: str = "Na"
    anion_label: str = "OH-"
    output_system_suffix: str = "slab"
    packmol_tolerance: float = 2.4
    packmol_seed: int = -1

    @property
    def resolved_xy_buffer(self) -> float:
        """Return the x/y boundary buffer."""
        return self.buffer / 2.0 if self.xy_buffer is None else self.xy_buffer

    @property
    def resolved_z_slab_buffer(self) -> float:
        """Return the z buffer around the slab surface."""
        return self.buffer if self.z_slab_buffer is None else self.z_slab_buffer

    @property
    def volume_correction(self) -> float:
        """Return the x/y volume correction used for density estimates."""
        return 2.0 * self.resolved_xy_buffer

    @property
    def resolved_z_shift(self) -> float:
        """Return the upper-region z shift applied when interlayer water is enabled."""
        if not self.add_interlayer_water:
            return 0.0
        if self.z_shift is not None:
            return self.z_shift
        return self.interlayer_thickness + self.buffer * 0.9

    @property
    def resolved_bubble_spacing(self) -> float:
        """Return center-to-center bubble spacing."""
        return self.bubble_spacing if self.bubble_spacing is not None else sum(self.gas_radii)

    @property
    def resolved_water_height_radius(self) -> float:
        """Return the radius reference used to set upper-water height."""
        return self.water_height_radius if self.water_height_radius is not None else self.gas_radii[0]

    @property
    def resolved_bubble_center_z_radius(self) -> float:
        """Return the radius reference used to set bubble z center."""
        if self.bubble_center_z_radius is not None:
            return self.bubble_center_z_radius
        return self.gas_radii[0]


@dataclass(frozen=True)
class SlabBuildConfig:
    """ASE slab construction settings."""

    bulk_structure: Path
    miller_index: Tuple[int, int, int] = (1, 0, 1)
    layers: int = 6
    vacuum: float = 100.0
    repeat: Tuple[int, int, int] = (11, 20, 1)
    atom_order: Tuple[str, ...] = ("O", "Ti")
    layer_tolerance: float = 0.1
    bottom_layer_index: int = 4
    slab_slice_margin: float = 0.5
    slab_name: str = "interface_slab"


@dataclass(frozen=True)
class PrebuiltInterfaceConfig:
    """Settings for using an already prepared interface structure."""

    interface_structure: Path
    layer_tolerance: float = 0.1
    bottom_layer_index: int = 0
    bottom_reference_z: Optional[float] = None
    top_reference_z: Optional[float] = None
    slab_name: str = "interface_slab"


@dataclass(frozen=True)
class DoubleBubbleSlabPlan:
    """Computed structure-preparation plan independent of file writing."""

    output_xyz_name: str
    lower_water: BoxRegion
    upper_water: BoxRegion
    interlayer_water: Optional[BoxRegion]
    bubble_constraints: Tuple[str, str]
    water_counts: Tuple[int, int, int]
    n2_counts: Tuple[int, int]
    ion_pair_count: int
    bubble_centers: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    bubble_volumes: Tuple[float, float]
    gas_volume: float

    @property
    def total_water_count(self) -> int:
        """Return total water count."""
        return sum(self.water_counts)

    @property
    def total_n2_count(self) -> int:
        """Return total N2 count."""
        return sum(self.n2_counts)


@dataclass(frozen=True)
class DoubleBubbleBuildResult:
    """Files and plan produced by a double-bubble slab build."""

    plan: DoubleBubbleSlabPlan
    output_dir: Path
    packmol_path: Path
    slab_xyz_path: Path
    poscar_path: Path
    slab_pdb_path: Path


def identify_z_layers(z_coordinates: Sequence[float], tolerance: float = 0.1) -> List[float]:
    """Group z coordinates into layer centers."""
    if len(z_coordinates) == 0:
        raise ValueError("At least one z coordinate is required")
    sorted_z = sorted(float(value) for value in z_coordinates)
    layers: List[float] = []
    current = [sorted_z[0]]
    for z_value in sorted_z[1:]:
        if abs(z_value - current[-1]) < tolerance:
            current.append(z_value)
        else:
            layers.append(float(np.mean(current)))
            current = [z_value]
    layers.append(float(np.mean(current)))
    return layers


def plan_double_bubble_slab(
    *,
    cell_lengths: Tuple[float, float, float],
    bottom_reference_z: float,
    top_reference_z: float,
    config: DoubleBubbleSlabConfig,
) -> DoubleBubbleSlabPlan:
    """Compute solvent regions, gas regions, and molecule counts."""
    if config.bubble_shape not in {"sphere", "cylinder"}:
        raise ValueError("bubble_shape must be 'sphere' or 'cylinder'")
    radius_a, radius_b = config.gas_radii
    if radius_a <= 0 or radius_b <= 0:
        raise ValueError("Gas bubble radii must be positive")

    cell_x, cell_y, _ = cell_lengths
    xy_buffer = config.resolved_xy_buffer
    z_slab_buffer = config.resolved_z_slab_buffer
    volume_correction = config.volume_correction

    lower_water = BoxRegion(
        xy_buffer,
        xy_buffer,
        bottom_reference_z - z_slab_buffer - config.interlayer_thickness,
        cell_x - xy_buffer,
        cell_y - xy_buffer,
        bottom_reference_z - z_slab_buffer,
    )

    interlayer_water = None
    if config.add_interlayer_water:
        interlayer_z_min = top_reference_z + z_slab_buffer
        interlayer_water = BoxRegion(
            xy_buffer,
            xy_buffer,
            interlayer_z_min,
            cell_x - xy_buffer,
            cell_y - xy_buffer,
            interlayer_z_min + config.interlayer_thickness,
        )

    z_offset = config.resolved_z_shift
    upper_water_z_min = top_reference_z + z_slab_buffer + z_offset
    upper_water = BoxRegion(
        xy_buffer,
        xy_buffer,
        upper_water_z_min,
        cell_x - xy_buffer,
        cell_y - xy_buffer,
        top_reference_z + config.resolved_water_height_radius * 3.5 + z_offset,
    )

    spacing = config.resolved_bubble_spacing
    center_a = (
        cell_x / 2.0 - spacing / 2.0,
        cell_y / 2.0,
        upper_water_z_min + config.resolved_bubble_center_z_radius,
    )
    center_b = (
        cell_x / 2.0 + spacing / 2.0,
        cell_y / 2.0,
        upper_water_z_min + config.resolved_bubble_center_z_radius,
    )

    if config.bubble_shape == "sphere":
        constraint_a = SphereRegion(*center_a, radius_a).packmol_region()
        constraint_b = SphereRegion(*center_b, radius_b).packmol_region()
        bubble_volume_a = sphere_volume(radius_a)
        bubble_volume_b = sphere_volume(radius_b)
        gas_overlap = two_sphere_intersection_volume(radius_a, radius_b, spacing)
    else:
        cylinder_y_start = upper_water.y_min
        cylinder_length = upper_water.y_max - upper_water.y_min
        constraint_a = CylinderRegion(
            center_a[0], cylinder_y_start, center_a[2], 0.0, 1.0, 0.0, radius_a, cylinder_length
        ).packmol_region()
        constraint_b = CylinderRegion(
            center_b[0], cylinder_y_start, center_b[2], 0.0, 1.0, 0.0, radius_b, cylinder_length
        ).packmol_region()
        cylinder_length_for_volume = cylinder_length + volume_correction
        bubble_volume_a = cylinder_volume(radius_a, cylinder_length_for_volume)
        bubble_volume_b = cylinder_volume(radius_b, cylinder_length_for_volume)
        gas_overlap = _parallel_cylinder_overlap_volume(
            radius_a, radius_b, spacing, cylinder_length_for_volume
        )

    gas_volume = bubble_volume_a + bubble_volume_b - gas_overlap
    n2_a = molecule_count_from_density(
        bubble_volume_a, config.n2_density_kg_m3, config.n2_molar_mass_g_mol
    )
    n2_b = molecule_count_from_density(
        bubble_volume_b, config.n2_density_kg_m3, config.n2_molar_mass_g_mol
    )

    lower_water_count = molecule_count_from_density(
        lower_water.buffered_xy_volume(volume_correction),
        config.water_density_kg_m3,
        config.water_molar_mass_g_mol,
    )
    upper_water_volume = upper_water.buffered_xy_volume(volume_correction) - gas_volume
    if upper_water_volume <= 0:
        raise ValueError("Upper water volume is not positive after gas-volume subtraction")
    upper_water_count = molecule_count_from_density(
        upper_water_volume, config.water_density_kg_m3, config.water_molar_mass_g_mol
    )
    interlayer_water_count = 0
    if interlayer_water is not None:
        interlayer_water_count = molecule_count_from_density(
            interlayer_water.buffered_xy_volume(volume_correction),
            config.water_density_kg_m3,
            config.water_molar_mass_g_mol,
        )

    ion_pair_count = 0
    if config.target_ph is not None:
        hydroxide_mol_l = 10 ** (config.target_ph - 14.0)
        ion_pair_count = int(math.ceil(hydroxide_mol_l * upper_water_volume * 1e-27 * 6.02214076e23))

    total_water = lower_water_count + upper_water_count + interlayer_water_count
    shape_label = "sphere" if config.bubble_shape == "sphere" else "cyl"
    ph_label = "" if config.target_ph is None else f"_ph{config.target_ph:g}"
    output_xyz_name = (
        f"{total_water}h2o_{n2_a + n2_b}n2_{shape_label}{ph_label}_{config.output_system_suffix}.xyz"
    )

    return DoubleBubbleSlabPlan(
        output_xyz_name=output_xyz_name,
        lower_water=lower_water,
        upper_water=upper_water,
        interlayer_water=interlayer_water,
        bubble_constraints=(constraint_a, constraint_b),
        water_counts=(lower_water_count, interlayer_water_count, upper_water_count),
        n2_counts=(n2_a, n2_b),
        ion_pair_count=ion_pair_count,
        bubble_centers=(center_a, center_b),
        bubble_volumes=(bubble_volume_a, bubble_volume_b),
        gas_volume=gas_volume,
    )


def build_packmol_input(
    *,
    plan: DoubleBubbleSlabPlan,
    slab_xyz_path: Path,
    templates: MoleculeTemplates,
    config: DoubleBubbleSlabConfig,
) -> PackmolInput:
    """Build a PACKMOL input object from a double-bubble plan."""
    lower_count, interlayer_count, upper_count = plan.water_counts
    n2_a, n2_b = plan.n2_counts
    constraint_a, constraint_b = plan.bubble_constraints
    structures: List[PackmolStructure] = [
        PackmolStructure(
            path=slab_xyz_path,
            number=1,
            fixed="0. 0. 0. 0. 0. 0.",
            comment="Fixed interface slab",
        ),
        PackmolStructure(
            path=templates.water,
            number=lower_count,
            constraints=[f"inside {plan.lower_water.packmol_box()}"],
            comment="Lower water region",
        ),
    ]

    if plan.interlayer_water is not None and interlayer_count > 0:
        structures.append(
            PackmolStructure(
                path=templates.water,
                number=interlayer_count,
                constraints=[f"inside {plan.interlayer_water.packmol_box()}"],
                comment="Interlayer water region",
            )
        )

    structures.append(
        PackmolStructure(
            path=templates.water,
            number=upper_count,
            constraints=[
                f"inside {plan.upper_water.packmol_box()}",
                f"outside {constraint_a}",
                f"outside {constraint_b}",
            ],
            comment="Upper water region excluding gas bubbles",
        )
    )

    if plan.ion_pair_count > 0:
        if templates.cation is None or templates.anion is None:
            raise ValueError("Cation and anion templates are required when target_ph creates ions")
        for template, label in [
            (templates.cation, config.cation_label),
            (templates.anion, config.anion_label),
        ]:
            structures.append(
                PackmolStructure(
                    path=template,
                    number=plan.ion_pair_count,
                    constraints=[
                        f"inside {plan.upper_water.packmol_box()}",
                        f"outside {constraint_a}",
                        f"outside {constraint_b}",
                    ],
                    comment=f"{label} ions in the upper water region",
                )
            )

    structures.extend(
        [
            PackmolStructure(
                path=templates.nitrogen,
                number=n2_a,
                constraints=[f"inside {constraint_a}"],
                comment="Gas bubble A",
            ),
            PackmolStructure(
                path=templates.nitrogen,
                number=n2_b,
                constraints=[f"inside {constraint_b}"],
                comment="Gas bubble B",
            ),
        ]
    )

    return PackmolInput(
        output_xyz=plan.output_xyz_name,
        structures=structures,
        tolerance=config.packmol_tolerance,
        seed=config.packmol_seed,
    )


def build_tio2_double_bubble_inputs(
    *,
    slab_config: SlabBuildConfig,
    bubble_config: DoubleBubbleSlabConfig,
    templates: MoleculeTemplates,
    output_dir: PathLike,
) -> DoubleBubbleBuildResult:
    """Build TiO2 interface files and a PACKMOL input for a double-bubble system."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    validate_template_paths(required_template_paths(templates, bubble_config))

    slab, layers = build_interface_slab(slab_config)
    bottom_reference_z = layers[slab_config.bottom_layer_index]
    top_reference_z = layers[-1]
    lengths = tuple(float(value) for value in slab.get_cell().lengths())

    plan = plan_double_bubble_slab(
        cell_lengths=(lengths[0], lengths[1], lengths[2]),
        bottom_reference_z=bottom_reference_z,
        top_reference_z=top_reference_z,
        config=bubble_config,
    )

    slab_xyz_path = output_path / f"{slab_config.slab_name}.xyz"
    poscar_path = output_path / "POSCAR"
    slab_pdb_path = output_path / f"{slab_config.slab_name}.pdb"
    packmol_path = output_path / "packmol.in"

    write_structure_files(slab, slab_xyz_path, poscar_path, slab_pdb_path)
    packmol_input = build_packmol_input(
        plan=plan,
        slab_xyz_path=slab_xyz_path,
        templates=templates,
        config=bubble_config,
    )
    write_packmol_input(packmol_input, packmol_path)

    return DoubleBubbleBuildResult(
        plan=plan,
        output_dir=output_path,
        packmol_path=packmol_path,
        slab_xyz_path=slab_xyz_path,
        poscar_path=poscar_path,
        slab_pdb_path=slab_pdb_path,
    )


def build_prebuilt_double_bubble_inputs(
    *,
    interface_config: PrebuiltInterfaceConfig,
    bubble_config: DoubleBubbleSlabConfig,
    templates: MoleculeTemplates,
    output_dir: PathLike,
) -> DoubleBubbleBuildResult:
    """Build PACKMOL input from an already prepared interface structure.

    This adapter does not construct a surface or assume a specific material.  It
    only reads a user-provided interface structure with ASE, infers or uses the
    supplied z references, writes normalized interface files, and reuses the
    shared double-bubble planning and PACKMOL rendering logic.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    validate_template_paths(required_template_paths(templates, bubble_config))

    interface, layers = read_prebuilt_interface(interface_config)
    bottom_reference_z = (
        interface_config.bottom_reference_z
        if interface_config.bottom_reference_z is not None
        else layers[interface_config.bottom_layer_index]
    )
    top_reference_z = (
        interface_config.top_reference_z
        if interface_config.top_reference_z is not None
        else layers[-1]
    )
    lengths = tuple(float(value) for value in interface.get_cell().lengths())
    plan = plan_double_bubble_slab(
        cell_lengths=(lengths[0], lengths[1], lengths[2]),
        bottom_reference_z=bottom_reference_z,
        top_reference_z=top_reference_z,
        config=bubble_config,
    )

    slab_xyz_path = output_path / f"{interface_config.slab_name}.xyz"
    poscar_path = output_path / "POSCAR"
    slab_pdb_path = output_path / f"{interface_config.slab_name}.pdb"
    packmol_path = output_path / "packmol.in"

    write_structure_files(interface, slab_xyz_path, poscar_path, slab_pdb_path)
    packmol_input = build_packmol_input(
        plan=plan,
        slab_xyz_path=slab_xyz_path,
        templates=templates,
        config=bubble_config,
    )
    write_packmol_input(packmol_input, packmol_path)

    return DoubleBubbleBuildResult(
        plan=plan,
        output_dir=output_path,
        packmol_path=packmol_path,
        slab_xyz_path=slab_xyz_path,
        poscar_path=poscar_path,
        slab_pdb_path=slab_pdb_path,
    )


def build_interface_slab(slab_config: SlabBuildConfig):
    """Build and slice an interface slab using ASE."""
    try:
        from ase import Atoms
        from ase.build import surface
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required for slab construction. Install molsimflow[structure].") from exc

    bulk = read(slab_config.bulk_structure)
    slab = surface(bulk, slab_config.miller_index, layers=slab_config.layers, vacuum=slab_config.vacuum)
    slab = slab.repeat(slab_config.repeat)
    slab.center(axis=2)

    symbols = slab.get_chemical_symbols()
    positions = slab.get_positions()
    order_map = {symbol: rank for rank, symbol in enumerate(slab_config.atom_order)}
    sorted_indices = sorted(range(len(symbols)), key=lambda index: order_map.get(symbols[index], len(order_map)))
    ordered_slab = Atoms(
        symbols=[symbols[index] for index in sorted_indices],
        positions=[positions[index] for index in sorted_indices],
        cell=slab.get_cell(),
        pbc=[True, True, False],
    )

    z_coordinates = ordered_slab.get_positions()[:, 2]
    layers = identify_z_layers(z_coordinates, slab_config.layer_tolerance)
    if len(layers) <= slab_config.bottom_layer_index:
        raise ValueError(
            f"Detected {len(layers)} layers; bottom_layer_index={slab_config.bottom_layer_index} is invalid"
        )

    z_min = layers[slab_config.bottom_layer_index] + slab_config.slab_slice_margin
    z_max = layers[-1] - slab_config.slab_slice_margin
    interface_indices = [
        index for index, z_value in enumerate(z_coordinates) if z_min <= z_value <= z_max
    ]
    if not interface_indices:
        raise ValueError("Interface slice selected no atoms")

    interface_slab = Atoms(
        symbols=[ordered_slab.symbols[index] for index in interface_indices],
        positions=[ordered_slab.positions[index] for index in interface_indices],
        cell=ordered_slab.get_cell(),
        pbc=[True, True, False],
    )
    return interface_slab, layers


def read_prebuilt_interface(interface_config: PrebuiltInterfaceConfig):
    """Read a prebuilt interface structure and infer z layers with ASE."""
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required for prebuilt interface reading. Install molsimflow[structure].") from exc

    interface = read(interface_config.interface_structure)
    lengths = tuple(float(value) for value in interface.get_cell().lengths())
    if lengths[0] <= 0 or lengths[1] <= 0 or lengths[2] <= 0:
        raise ValueError("Prebuilt interface structure must contain non-zero cell lengths")

    z_coordinates = interface.get_positions()[:, 2]
    layers = identify_z_layers(z_coordinates, interface_config.layer_tolerance)
    if interface_config.bottom_reference_z is None and len(layers) <= interface_config.bottom_layer_index:
        raise ValueError(
            "Detected {} layers; bottom_layer_index={} is invalid".format(
                len(layers), interface_config.bottom_layer_index
            )
        )
    return interface, layers


def write_structure_files(slab, xyz_path: Path, poscar_path: Path, pdb_path: Path) -> None:
    """Write interface slab files using ASE."""
    try:
        from ase.io import write
    except ImportError as exc:
        raise ImportError("ASE is required for structure writing. Install molsimflow[structure].") from exc
    write(xyz_path, slab)
    write(poscar_path, slab, vasp5=True, sort=False, direct=False)
    write(pdb_path, slab)


def required_template_paths(
    templates: MoleculeTemplates, config: DoubleBubbleSlabConfig
) -> List[Path]:
    """Return molecule template paths required by a build config."""
    required = [templates.water, templates.nitrogen]
    if config.target_ph is not None:
        if templates.cation is None or templates.anion is None:
            raise ValueError("Cation and anion templates are required when target_ph is enabled")
        required.extend([templates.cation, templates.anion])
    return required


def resolve_molecule_templates(
    *,
    molecule_dir: Optional[Path],
    water_xyz: Optional[Path],
    n2_xyz: Optional[Path],
    cation_xyz: Optional[Path],
    anion_xyz: Optional[Path],
) -> MoleculeTemplates:
    """Resolve molecule template paths from explicit files or a molecule directory."""
    return MoleculeTemplates(
        water=resolve_template_path(water_xyz, molecule_dir, "H2O.xyz"),
        nitrogen=resolve_template_path(n2_xyz, molecule_dir, "N2.xyz"),
        cation=resolve_template_path(cation_xyz, molecule_dir, "Na.xyz")
        if cation_xyz is not None or molecule_dir is not None
        else None,
        anion=resolve_template_path(anion_xyz, molecule_dir, "OH-.xyz")
        if anion_xyz is not None or molecule_dir is not None
        else None,
    )


def _parallel_cylinder_overlap_volume(radius_a: float, radius_b: float, distance: float, length: float) -> float:
    """Approximate overlap volume for equal-axis cylinders using the smaller radius."""
    if distance >= radius_a + radius_b:
        return 0.0
    radius = min(radius_a, radius_b)
    if distance >= 2.0 * radius:
        return 0.0
    if distance <= 0.0:
        return math.pi * radius**2 * length
    theta = 2.0 * math.acos(distance / (2.0 * radius))
    area_overlap = radius**2 * (theta - math.sin(theta))
    return area_overlap * length
