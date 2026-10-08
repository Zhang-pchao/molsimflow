from molsimflow.postprocess.constant_force_stage_b_synthesis import (
    classify_exchange,
    classify_layer_response,
    classify_tpcl,
)


def test_exchange_classification_requires_material_same_sign_fraction():
    rows = [
        {
            "category": "PERSISTENT_ISLAND_TRANSFER",
            "signed_fraction_of_total_response": "0.6",
            "response_velocity_mps": "0.6",
            "total_response_velocity_mps": "1.0",
        },
        {
            "category": "PERSISTENT_ISLAND_TRANSFER",
            "signed_fraction_of_total_response": "0.7",
            "response_velocity_mps": "-0.7",
            "total_response_velocity_mps": "-1.0",
        },
    ]
    decision, maximum, _ = classify_exchange(rows)
    assert decision == "EXCHANGE_DOMINATED_CANDIDATE"
    assert maximum == 0.7


def test_exchange_classification_rejects_small_opposite_contribution():
    rows = [
        {
            "category": "PERSISTENT_ISLAND_TRANSFER",
            "signed_fraction_of_total_response": "-0.01",
            "response_velocity_mps": "-0.01",
            "total_response_velocity_mps": "1.0",
        },
        {
            "category": "PERSISTENT_ISLAND_TRANSFER",
            "signed_fraction_of_total_response": "0.02",
            "response_velocity_mps": "0.02",
            "total_response_velocity_mps": "1.0",
        },
    ]
    decision, _, _ = classify_exchange(rows)
    assert decision == "EXCHANGE_CONTRIBUTION_LIMITED"


def test_layer_classification_finds_persistent_opposite_pair():
    rows = []
    for block in range(80):
        for layer, value in ((1, 2.0), (2, -3.0), (3, 1.0 if block < 40 else -1.0)):
            rows.append(
                {
                    "branch_id": "f8e-5_y",
                    "layer_index": str(layer),
                    "start_ps": str(block * 50.0),
                    "mean_count": "10",
                    "molecule_samples": "50",
                    "raw_axis_velocity_mps": str(value),
                }
            )
    decision, detail = classify_layer_response(rows)
    assert decision == "TIME_DEPENDENT_LAYER_OPPOSED_Y_CANDIDATE"
    assert "layers 1 and 2" in detail


def test_tpcl_classification_uses_matched_anchor_retention_column():
    anchor_summary = [
        {
            "case_id": "mixed291",
            "mean_anchor_pair_retained_fraction": "0.6",
        },
        {
            "case_id": "mixed291",
            "mean_anchor_pair_retained_fraction": "0.7",
        },
    ]
    matched_rows = [
        {
            "case_id": "mixed291",
            "event_minus_control_delta_anchor_anchor_pair_retained_fraction": "0.1",
        },
        {
            "case_id": "mixed291",
            "event_minus_control_delta_anchor_anchor_pair_retained_fraction": "-0.2",
        },
        {
            "case_id": "ch3_only",
            "event_minus_control_delta_anchor_anchor_pair_retained_fraction": "nan",
        },
    ]
    decision, detail = classify_tpcl(anchor_summary, matched_rows)
    assert decision == "DYNAMIC_ANCHOR_ASSOCIATION_ONLY"
