import math

from molsimflow.postprocess.reaction_kinetics import (
    GAS_CONSTANT_KJ_MOL_K,
    PathwayBarrier,
    build_barrier_sensitivity,
    build_two_channel_competition,
    eyring_rate,
)


def test_eyring_sensitivity_and_conditional_competition_are_temperature_configurable():
    temperature = 310.0
    baseline = eyring_rate(75.0, temperature_K=temperature)
    lowered = eyring_rate(70.0, temperature_K=temperature)
    assert math.isclose(
        lowered / baseline,
        math.exp(5.0 / (GAS_CONSTANT_KJ_MOL_K * temperature)),
        rel_tol=1.0e-12,
    )

    pathway = PathwayBarrier("example pathway", 75.0)
    sensitivity = build_barrier_sensitivity([pathway], [-5.0, 0.0], temperature_K=temperature)
    competition = build_two_channel_competition(
        [pathway], [1.0], temperature_K=temperature
    )

    assert [row["barrier_shift_kj_mol"] for row in sensitivity] == [-5.0, 0.0]
    expected = competition[0]["pathway_rate_s_inv"] / (
        competition[0]["pathway_rate_s_inv"] + 1.0
    )
    assert competition[0]["conditional_pathway_fraction"] == expected
