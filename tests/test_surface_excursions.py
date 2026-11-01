import numpy as np
import pytest
from molsimflow.postprocess.surface_excursions import observed_exterior_return as path

def test_return_and_brackets():
    x=path([1,0,0,1],[0,0,1,0],[0,.5,1,1.5],.5)
    assert x['status']=='VISITED_AND_RETURNED'
    assert (x['visit_lower_ps'],x['visit_upper_ps'])==(.5,1)
    assert (x['return_lower_ps'],x['return_upper_ps'])==(1,1.5)

def test_label_recrossing_is_not_exterior_visit():
    x=path([1,0,1,1],[0,0,0,0],np.arange(4),1)
    assert x['label_return_observed'] and not x['exterior_visit_observed']

def test_terminal_visit_censors_return():
    assert path([1,0],[0,1],[0,1],1)['status']=='VISITED_RETURN_RIGHT_CENSORED'

def test_thinning_can_lose_visit():
    a=np.array([1,0,1,1,1]);b=np.array([0,1,0,0,0]);t=np.arange(5)
    assert path(a,b,t,1)['exterior_return_observed']
    assert not path(a[::2],b[::2],t[::2],2)['exterior_visit_observed']

@pytest.mark.parametrize('a,b,t', [([1,1],[0,1],[0,1]),([1,-1],[0,0],[0,1]),([1,0],[0,1],[0,2]),([1,0],[0,np.nan],[0,1])])
def test_invalid_paths_fail_closed(a,b,t):
    with pytest.raises(ValueError):path(a,b,t,1)
