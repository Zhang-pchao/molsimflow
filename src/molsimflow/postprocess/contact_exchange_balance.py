"""Exact contact-population bookkeeping and minimal conditional baselines.

States: 0 outside tracked main component; 1 main/far; 2 main/near but not
contacting; 3 main/contacting. Sampling interval determines transition
probabilities; these are not continuous-time rate constants.
"""
import numpy as np

def population_states(main: np.ndarray, contact: np.ndarray,
                      distance: np.ndarray, *, near_cutoff: float=8.5) -> np.ndarray:
    main=np.asarray(main,bool);contact=np.asarray(contact,bool);distance=np.asarray(distance,float)
    if main.shape!=contact.shape or main.shape!=distance.shape or main.ndim!=2:
        raise ValueError("Aligned frame by molecule arrays required")
    if np.any(contact&~main) or not np.isfinite(distance).all() or near_cutoff<=0:
        raise ValueError("Finite distance, positive cutoff and main-only contact required")
    states=np.zeros(main.shape,np.int8)
    states[main]=1;states[main&(distance<=near_cutoff)]=2;states[contact]=3
    return states

def contact_exchange(states: np.ndarray) -> dict[str,np.ndarray]:
    """Return all transition channels and exactly closing contact balances."""
    states=np.asarray(states)
    if states.ndim!=2 or len(states)<2 or np.any((states<0)|(states>3)):
        raise ValueError("At least two frames of states 0..3 required")
    n=len(states)-1
    code=(16*np.arange(n)[:,None]+4*states[:-1]+states[1:]).ravel()
    flow=np.bincount(code,minlength=16*n).reshape(n,4,4)
    population=np.stack([(states==k).sum(axis=1) for k in range(4)],axis=1)
    gain=flow[:,1,3]+flow[:,2,3]
    loss=flow[:,3,1]+flow[:,3,2]
    member_in=flow[:,0,3];member_out=flow[:,3,0]
    delta=np.diff(population[:,3])
    if not np.array_equal(delta,gain-loss+member_in-member_out):
        raise AssertionError("Contact-population balance did not close")
    if not np.array_equal(flow.sum(axis=2),population[:-1]):
        raise AssertionError("Transition origins do not match population")
    if not np.array_equal(flow.sum(axis=1),population[1:]):
        raise AssertionError("Transition destinations do not match population")
    return dict(flow=flow,population=population,gain=gain,loss=loss,
                member_in=member_in,member_out=member_out,delta=delta)

def fit_contact_probabilities(flow: np.ndarray, *, near_only: bool) -> dict:
    """Fit two empirical per-sample probabilities on supplied training counts."""
    flow=np.asarray(flow)
    if flow.ndim!=3 or flow.shape[1:]!=(4,4) or np.any(flow<0):
        raise ValueError("Nonnegative interval transition counts required")
    total=flow.sum(axis=0)
    exposure=total.sum(axis=1)
    source=[2] if near_only else [1,2]
    eon=int(exposure[source].sum());eoff=int(exposure[3])
    on=int(total[source,3].sum());off=int(total[3,1:3].sum())
    return dict(p_on=None if not eon else on/eon,p_off=None if not eoff else off/eoff,
                on_events=on,off_events=off,on_exposure=eon,off_exposure=eoff,
                far_to_contact_events=int(total[1,3]),near_only=near_only)

def predict_contact_increment(population: np.ndarray, fit: dict) -> np.ndarray:
    """Predict conditional one-step contact change excluding membership flux.

    Uses observed current state populations. This is not a free-running
    prediction of bubble motion or a new macroscopic constitutive law.
    """
    pop=np.asarray(population,float)
    if pop.ndim!=2 or pop.shape[1]!=4 or np.any(pop<0):
        raise ValueError("Four nonnegative populations per time interval required")
    if fit["p_on"] is None or fit["p_off"] is None:
        return np.full(len(pop),np.nan)
    available=pop[:,2] if fit["near_only"] else pop[:,1]+pop[:,2]
    return fit["p_on"]*available-fit["p_off"]*pop[:,3]
