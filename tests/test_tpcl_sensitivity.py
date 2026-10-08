import json

from molsimflow.postprocess.tpcl_sensitivity import load_sources, read_sources


def test_load_sources_requires_accepted_matching_results(tmp_path):
    root = tmp_path / "analysis"
    run = root / "run" / "42"
    (run / "results").mkdir(parents=True)
    (run / "inputs").mkdir()
    (run / "ANALYSIS-RESULT.txt").write_text("status=PASS\n", encoding="utf-8")
    (run / "results" / "summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "raw_frames": 10,
                "valid_contour_frames": 10,
                "contour_valid_fraction": 1.0,
                "candidate_event_count": 0,
                "candidate_arc_record_count": 0,
                "scientific_classification": "NO_CANDIDATE",
            }
        ),
        encoding="utf-8",
    )
    (run / "inputs" / "config.json").write_text(
        json.dumps({"contact_cutoff_A": 3.5, "arc_bins": 36}), encoding="utf-8"
    )
    latest = root / "latest" / "stage" / "case"
    latest.parent.mkdir(parents=True)
    latest.symlink_to(run)
    manifest = tmp_path / "sources.tsv"
    manifest.write_text(
        "parameter_id\tcase_id\tcontact_cutoff_A\tarc_bins\tsource_stage\n"
        "p\tcase\t3.5\t36\tstage\n",
        encoding="utf-8",
    )

    rows = load_sources(read_sources(manifest), root)
    assert rows[0]["job_id"] == "42"
    assert rows[0]["candidate_event_count"] == 0
