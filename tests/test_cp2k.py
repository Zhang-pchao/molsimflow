import numpy as np

from molsimflow.io.cp2k import HARTREE_PER_BOHR_TO_EV_PER_A, parse_cp2k_energy_forces


def test_cp2k_parser_uses_final_complete_energy_force_evaluation(tmp_path):
    output = tmp_path / "energy_force.out"
    output.write_text(
        "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -1.0\n"
        "ATOMIC FORCES in [a.u.]\n"
        " 1 1 H 0.0 0.0 0.0\n"
        " SUM OF ATOMIC FORCES 0.0 0.0 0.0\n"
        "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -1.25D+00\n"
        "ATOMIC FORCES in [a.u.]\n"
        " 1 1 H 0.1 -0.2 0.0\n"
        " 2 2 O -0.01 0.02 0.03\n"
        " SUM OF ATOMIC FORCES 0.0 0.0 0.0\n"
        "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -2.0\n"
        "ATOMIC FORCES in [a.u.]\n"
        " 1 1 H 9.0 9.0 9.0\n",
        encoding="utf-8",
    )

    result = parse_cp2k_energy_forces(output, atom_count=2)

    assert result.energy_hartree == -1.25
    assert np.allclose(
        result.forces_eV_A,
        np.asarray([[0.1, -0.2, 0.0], [-0.01, 0.02, 0.03]])
        * HARTREE_PER_BOHR_TO_EV_PER_A,
    )
