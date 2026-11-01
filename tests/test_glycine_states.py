import numpy as np
import pytest
from molsimflow.postprocess.glycine_states import classify_glycine_states, state_path_fractions


def test_stoichiometry_counterions_and_shared_protons():
    # Two waters followed by N and two carboxyl O centers.
    hard=np.array([[2,2,2,1,0],[2,2,3,0,0],[3,2,2,0,0],[1,2,3,1,0],[3,1,2,1,0]])
    soft=hard.astype(float)
    result=classify_glycine_states(soft,hard)
    assert result['core_labels'].tolist()==[0,1,2,3,4]
    soft[2,0]-=.4;soft[2,2]+=.4
    result=classify_glycine_states(soft,hard)
    assert result['hard_labels'][2]==2 and result['core_labels'][2]==4


def test_correlated_bead_fraction_and_missing_state():
    result=state_path_fractions(np.array([[0,0,1,1],[1,1,1,1]]))
    np.testing.assert_allclose(result,[[.5,.5,0,0,0],[0,1,0,0,0]])
    assert state_path_fractions(np.empty((0,16),dtype=int)).shape==(0,5)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):classify_glycine_states([np.nan]*5,[2,2,2,1,0])
    with pytest.raises(ValueError):classify_glycine_states([2]*5,[2]*5,tolerance=.8)
