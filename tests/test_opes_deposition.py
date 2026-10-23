"""Focused parser and path-versus-local OPES deposition checks."""

from __future__ import annotations

import numpy as np
import pytest

from molsimflow.postprocess.opes_deposition import (
    compare_opes_deposition_weights,
    read_opes_kernel_batches,
)


def test_selected_batches_and_permuted_bead_matching(tmp_path):
    kernels = tmp_path / "KERNELS"
    kernels.write_text(
        "#! FIELDS time x y sigma_x sigma_y height logweight\n"
        "#! SET action OPES_METAD_kernels\n"
        "125 2 4 0.1 0.2 1.0 4\n"
        "125 1 3 0.1 0.2 1.0 2\n"
        "250 5 6 0.1 0.2 1.0 3\n"
        "250 7 8 0.1 0.2 1.0 5\n"
    )
    first, second = read_opes_kernel_batches(kernels, [125, 250], expected_rows_per_time=2)
    assert first.cv_names == ("x", "y")
    assert second.time == 250
    checked = compare_opes_deposition_weights(
        first,
        np.array([[1.0, 3.0], [2.0, 4.0]]),
        np.array([0.2, 0.4]),
        common_path_bias_eV=0.3,
        kbt_eV=0.1,
    )
    assert np.array_equal(checked.logged_by_bead, [2.0, 4.0])
    assert checked.maximum_cv_match_error == 0
    assert checked.maximum_local_logweight_error < 1e-14
    assert checked.maximum_common_path_logweight_error == pytest.approx(1.0)


def test_reject_missing_time_and_invalid_batch(tmp_path):
    kernels = tmp_path / "KERNELS"
    kernels.write_text("#! FIELDS time x sigma_x height logweight\n125 1 0.1 1.0 2\n")
    with pytest.raises(ValueError, match="missing"):
        read_opes_kernel_batches(kernels, [125, 250])
    with pytest.raises(ValueError, match="row count"):
        read_opes_kernel_batches(kernels, [125], expected_rows_per_time=2)


def test_reject_nonbijective_centers(tmp_path):
    kernels = tmp_path / "KERNELS"
    kernels.write_text(
        "#! FIELDS time x sigma_x height logweight\n125 1 0.1 1.0 2\n125 1 0.1 1.0 2\n"
    )
    (batch,) = read_opes_kernel_batches(kernels, [125], expected_rows_per_time=2)
    with pytest.raises(ValueError, match="bijectively"):
        compare_opes_deposition_weights(
            batch,
            np.array([[1.0], [2.0]]),
            np.array([0.2, 0.4]),
            common_path_bias_eV=0.3,
            kbt_eV=0.1,
        )
