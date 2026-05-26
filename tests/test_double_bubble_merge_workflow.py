from molsimflow.workflows.double_bubble_merge import recommended_postprocess_stages


def test_double_bubble_merge_stage_plan_has_migrated_core_stages():
    stages = recommended_postprocess_stages()
    by_name = {stage.name: stage for stage in stages}

    assert by_name["coalescence_state"].status == "migrated"
    assert by_name["bridge_descriptors"].status == "partial"
    assert "molsimflow.postprocess.coalescence_state" in by_name["coalescence_state"].reusable_module
