from molsimflow.workflows.double_bubble_merge import (
    recommended_postprocess_stages,
    residual_adapter_plan,
)


def test_double_bubble_merge_stage_plan_has_migrated_core_stages():
    stages = recommended_postprocess_stages()
    by_name = {stage.name: stage for stage in stages}

    assert by_name["coalescence_state"].status == "migrated"
    assert by_name["bridge_descriptors"].status == "migrated"
    assert by_name["bridge_water_escape"].status == "migrated"
    assert by_name["hbond_network"].status == "migrated"
    assert by_name["contact_graph"].status == "migrated"
    assert by_name["local_environment"].status == "migrated"
    assert "molsimflow.postprocess.coalescence_state" in by_name["coalescence_state"].reusable_module


def test_double_bubble_merge_residual_adapters_are_optional_or_rejected():
    adapters = residual_adapter_plan()
    by_name = {adapter.name: adapter for adapter in adapters}

    assert by_name["seed_position_table_from_trajectory"].status == "optional_workflow_adapter"
    assert by_name["hbond_edges_from_trajectory"].expected_output.endswith(
        "molsimflow.postprocess.hbond_network"
    )
    assert by_name["publication_and_case_synthesis"].status == "do_not_migrate_directly"
