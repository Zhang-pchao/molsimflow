"""Classification guard for coarse front/rear block modes."""

from molsimflow.postprocess.nanobubble_ion_edge_blocks import motion_mode


def test_sign_modes_do_not_imply_depinning():
    assert motion_mode(2, 1) == "both_forward"
    assert motion_mode(-2, -1) == "both_reverse"
    assert motion_mode(2, -1) == "width_expanding_or_mixed"
    assert motion_mode(-2, 1) == "width_contracting_or_mixed"
