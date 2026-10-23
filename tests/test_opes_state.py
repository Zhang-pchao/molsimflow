from pathlib import Path

import numpy as np
import pytest

from molsimflow.postprocess.opes_state import (
    evaluate_nonperiodic_opes_state,
    read_opes_states,
)


def _state_text(time: int) -> str:
    return (
        "#! FIELDS time cv1 cv2 sigma_cv1 sigma_cv2 height\n"
        "#! SET action OPES_METAD_state\n"
        "#! SET biasfactor 20\n"
        "#! SET epsilon 1e-6\n"
        "#! SET kernel_cutoff 3\n"
        "#! SET zed 0.5\n"
        "#! SET sum_weights 5\n"
        "#! SET counter 3\n"
        f"{time} 0 0 1 2 2\n"
        f"{time} 1 1 2 1 3\n"
    )


def test_state_blocks_and_nonperiodic_field(tmp_path: Path) -> None:
    source = tmp_path / "STATE"
    source.write_text(_state_text(100) + _state_text(200), encoding="utf-8")
    states = read_opes_states(source)
    assert [state.time_fs for state in states] == [100, 200]
    assert states[0].cv_names == ("cv1", "cv2")
    points = np.array([[0.3, -0.2], [0.8, 0.1]])
    energy, force = evaluate_nonperiodic_opes_state(states[0], points, kbt_eV=0.025)
    r2_0 = 0.3**2 + (-0.2 / 2) ** 2
    r2_1 = ((0.3 - 1) / 2) ** 2 + (-0.2 - 1) ** 2
    probability = (
        2 * (np.exp(-r2_0 / 2) - np.exp(-4.5))
        + 3 * (np.exp(-r2_1 / 2) - np.exp(-4.5))
    ) / 5
    expected = 0.025 * (1 - 1 / 20) * np.log(probability / 0.5 + 1e-6)
    assert energy[0] == pytest.approx(expected, abs=1e-15)
    step = 1e-6
    for axis in range(2):
        up = points[:1].copy()
        down = points[:1].copy()
        up[0, axis] += step
        down[0, axis] -= step
        e_up = evaluate_nonperiodic_opes_state(states[0], up, kbt_eV=0.025)[0][0]
        e_down = evaluate_nonperiodic_opes_state(states[0], down, kbt_eV=0.025)[0][0]
        assert force[0, axis] == pytest.approx(-(e_up - e_down) / (2 * step), abs=1e-10)
    assert np.isfinite(energy).all()
    assert np.isfinite(force).all()


def test_cutoff_gives_epsilon_plateau(tmp_path: Path) -> None:
    source = tmp_path / "STATE"
    source.write_text(_state_text(100), encoding="utf-8")
    state = read_opes_states(source)[0]
    energy, force = evaluate_nonperiodic_opes_state(state, np.array([[100.0, 100.0]]), kbt_eV=0.025)
    assert energy[0] == pytest.approx(0.025 * 0.95 * np.log(1e-6))
    assert np.array_equal(force, np.zeros((1, 2)))


def test_reject_nonmonotonic_and_malformed_states(tmp_path: Path) -> None:
    source = tmp_path / "STATE"
    source.write_text(_state_text(100) + _state_text(100), encoding="utf-8")
    with pytest.raises(ValueError, match="increasing"):
        read_opes_states(source)
    source.write_text(_state_text(100).replace("sigma_cv1", "sigma_other"), encoding="utf-8")
    with pytest.raises(ValueError, match="match"):
        read_opes_states(source)
    source.write_text(_state_text(100).replace(" 0 0 1 2 2", " 0 0 0 2 2"), encoding="utf-8")
    with pytest.raises(ValueError, match="parameter or kernel"):
        read_opes_states(source)


def test_state_preserves_fractional_time_with_declared_scale(tmp_path):
    path = tmp_path / "STATE"
    path.write_text(_state_text(100).replace("100 ", "0.125 "))
    state = read_opes_states(path, time_scale_to_fs=1000.)[0]
    assert state.time_fs == 125.
