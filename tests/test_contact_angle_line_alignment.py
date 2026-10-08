from molsimflow.postprocess.contact_angle_line_alignment import align_blocks


def test_align_blocks_flags_constant_radius_angle_change():
    angles = [
        {"first_step": 0, "last_step": 2, "start_time_ns": 0, "end_time_ns": 1, "angle": 80},
        {"first_step": 3, "last_step": 5, "start_time_ns": 1, "end_time_ns": 2, "angle": 84},
    ]
    lines = [{"step": step, "radius": 10.0 + 0.1 * step} for step in range(6)]
    rows = align_blocks(
        angles,
        lines,
        configuration="baseline",
        angle_column="angle",
        radius_column="radius",
        radius_stability=1.0,
        angle_change=3.0,
    )
    assert rows[1]["pinning_candidate"] is True
    assert rows[1]["delta_contact_angle_deg"] == 4.0
