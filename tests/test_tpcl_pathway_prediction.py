import numpy as np

from molsimflow.postprocess.tpcl_pathway_prediction import (
    MODEL_FEATURES,
    RESPONSE_FEATURES,
    PathwayConfig,
    cluster_responses,
    deduplicate_risk_rows,
)


def test_surface_blind_clustering_and_risk_anchor_deduplication():
    event_rows = []
    for case_id, offset in (("a", 0.0), ("b", 100.0)):
        for index in range(20):
            low = index < 10
            row = {"case_id": case_id}
            for feature_index, field in enumerate(RESPONSE_FEATURES):
                row[field] = str(offset + (-2.0 if low else 2.0) + 0.01 * feature_index)
            event_rows.append(row)
    labels, _, _, summary = cluster_responses(
        event_rows,
        PathwayConfig(
            cluster_min=2,
            cluster_max=2,
            cluster_n_init=5,
            silhouette_gate=0.1,
            median_leave_one_feature_ari_gate=0.5,
        ),
    )
    assert summary["selected_cluster_count"] == 2
    assert summary["phenotype_stable"]
    assert len(np.unique(labels[:20])) == 2
    assert len(np.unique(labels[20:])) == 2

    features = MODEL_FEATURES[-1][1]
    base = {
        "case_id": "a",
        "primary_arc_index": "1",
        "sample_step": "10",
        "is_event": "0",
        "risk_set_weight": "0.1",
        **{field: "nan" for field in features},
    }
    rows, audit = deduplicate_risk_rows([base, {**base, "risk_set_weight": "0.2"}])
    assert len(rows) == 1
    assert np.isclose(rows[0]["risk_set_weight"], 0.3)
    assert audit["duplicate_anchor_key_count"] == 1
