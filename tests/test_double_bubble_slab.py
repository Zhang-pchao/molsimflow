from pathlib import Path

from molsimflow.structure.double_bubble_slab import (
    DoubleBubbleSlabConfig,
    MoleculeTemplates,
    build_packmol_input,
    identify_z_layers,
    plan_double_bubble_slab,
    required_template_paths,
)


def test_identify_z_layers_groups_close_coordinates():
    layers = identify_z_layers([0.0, 0.04, 1.0, 1.05, 2.0], tolerance=0.1)
    assert len(layers) == 3
    assert round(layers[0], 2) == 0.02


def test_plan_double_bubble_slab_uses_equal_volume_reference_options():
    config = DoubleBubbleSlabConfig(
        gas_radii=(16.87, 16.87),
        bubble_spacing=50.0,
        target_ph=13.4,
        water_height_radius=19.0,
        bubble_center_z_radius=19.0,
        output_system_suffix="tio2",
    )
    plan = plan_double_bubble_slab(
        cell_lengths=(80.0, 100.0, 120.0),
        bottom_reference_z=20.0,
        top_reference_z=40.0,
        config=config,
    )
    assert plan.n2_counts[0] == plan.n2_counts[1]
    assert plan.ion_pair_count > 0
    assert "sphere" in plan.bubble_constraints[0]
    assert plan.bubble_centers[0][2] == plan.upper_water.z_min + 19.0
    assert plan.output_xyz_name.endswith("_tio2.xyz")


def test_build_packmol_input_contains_reusable_template_paths():
    config = DoubleBubbleSlabConfig(gas_radii=(4.0, 3.0), target_ph=None)
    plan = plan_double_bubble_slab(
        cell_lengths=(40.0, 50.0, 60.0),
        bottom_reference_z=10.0,
        top_reference_z=20.0,
        config=config,
    )
    packmol_input = build_packmol_input(
        plan=plan,
        slab_xyz_path=Path("interface.xyz"),
        templates=MoleculeTemplates(water=Path("mol/H2O.xyz"), nitrogen=Path("mol/N2.xyz")),
        config=config,
    )
    text = packmol_input.render()
    assert "structure interface.xyz" in text
    assert "structure mol/H2O.xyz" in text
    assert "outside sphere" in text
    assert "Gas bubble A" in text


def test_required_template_paths_requires_ion_templates_when_ph_enabled():
    config = DoubleBubbleSlabConfig(target_ph=13.4)
    templates = MoleculeTemplates(water=Path("H2O.xyz"), nitrogen=Path("N2.xyz"))
    try:
        required_template_paths(templates, config)
    except ValueError as exc:
        assert "Cation and anion templates" in str(exc)
    else:
        raise AssertionError("Expected missing ion templates to raise ValueError")


def test_required_template_paths_allows_missing_ions_when_ph_disabled():
    config = DoubleBubbleSlabConfig(target_ph=None)
    templates = MoleculeTemplates(water=Path("H2O.xyz"), nitrogen=Path("N2.xyz"))
    assert required_template_paths(templates, config) == [Path("H2O.xyz"), Path("N2.xyz")]
