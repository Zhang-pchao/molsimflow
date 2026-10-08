from molsimflow.postprocess.tpcl_three_highlight_closure import decide_closure


def test_final_closure_preserves_causal_boundary():
    rows = decide_closure({"scientific_gate": "MECHANISM_NOT_ESTABLISHED_SINGLE_HISTORY"})
    by_id = {row["highlight_id"]: row for row in rows}
    assert by_id["H1"]["closure"] == "CLOSED_PUBLICATION_GRADE"
    assert by_id["H2"]["closure"] == "CLOSED_WITH_LIMITS"
    assert by_id["H3"]["closure"] == "MECHANISM_NOT_ESTABLISHED"
    assert by_id["H3"]["requires_new_md"]
    assert by_id["OVERALL"]["closure"] == "REQUIRES_NEW_MD"


def test_closure_does_not_call_single_history_independent():
    rows = decide_closure({})
    h3 = next(row for row in rows if row["highlight_id"] == "H3")
    assert h3["evidence_level"] == "single_high_cadence_parent_history_with_parameter_audit"
