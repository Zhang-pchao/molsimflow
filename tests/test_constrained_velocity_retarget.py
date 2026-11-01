import numpy as np
import pytest
from molsimflow.postprocess.constrained_velocity_retarget import retarget_linear_velocity_mode as retarget

def sample():
    rng = np.random.default_rng(274)
    x = rng.normal(size=(14, 3))
    v = rng.normal(size=(14, 3))
    m = rng.uniform(2, 30, 14)
    t = np.zeros_like(v); t[0] = [.3, .8, -.2]
    return x, v, m, t

def test_physical_invariants_and_actual_observable():
    x, v, m, t = sample()
    wanted = float(np.sum(t*v)) + .4
    a = retarget(v, m, x, t, wanted); d = a.velocities-v
    assert abs(np.sum(t*a.velocities)-wanted) < 1e-12
    np.testing.assert_allclose(np.sum(m[:, None]*d, axis=0), 0, atol=2e-12)
    np.testing.assert_allclose(np.sum(np.cross(x, m[:, None]*d), axis=0), 0, atol=2e-12)
    assert abs(np.sum(m[:, None]*(a.velocities**2-v**2))) < 2e-12

def test_identity_and_round_trip():
    x, v, m, t = sample(); original = float(np.sum(t*v))
    a = retarget(v, m, x, t, original)
    np.testing.assert_allclose(a.velocities, v, atol=1e-12)
    b = retarget(v, m, x, t, original+.5)
    c = retarget(b.velocities, m, x, t, original)
    np.testing.assert_allclose(c.velocities, v, atol=1e-12)

def test_boost_translation_and_rotation_covariance():
    x, v, m, t = sample(); wanted = float(np.sum(t*v))+.3
    a = retarget(v, m, x, t, wanted)
    boost = np.array([2., -3., 1.])
    b = retarget(v+boost, m, x+[30., -20., 10.], t, wanted+np.sum(t*boost))
    np.testing.assert_allclose(b.velocities-boost, a.velocities, atol=2e-12)
    Q = np.array([[0,1,0],[0,0,1],[1,0,0]])
    c = retarget(v@Q, m, x@Q, t@Q, wanted)
    np.testing.assert_allclose(c.velocities, a.velocities@Q, atol=2e-12)

def test_additional_constraints_and_scaling():
    x, v, m, t = sample(); extra = np.zeros((1, len(v), 3)); extra[0,1] = [1.,2.,3.]
    wanted = float(np.sum(t*v))+.2
    a = retarget(v, m, x, t, wanted, extra)
    assert abs(np.sum(extra*(a.velocities-v))) < 1e-12
    b = retarget(v, m, x, t, wanted, np.concatenate([extra, extra*1e8, extra*1e-8]))
    np.testing.assert_allclose(a.velocities, b.velocities, atol=2e-12)

def test_group_increment_preserves_internal_motion():
    x, v, m, t = sample(); a = retarget(v, m, x, t, float(np.sum(t*v))+.4)
    ma, mb = .7*m[0], .3*m[0]; internal = np.array([.4, -.3, .2])
    va = v[0]+internal; vb = v[0]-ma/mb*internal; dv = a.velocities[0]-v[0]
    np.testing.assert_allclose((va+dv)-(vb+dv), va-vb, atol=1e-12)
    ki0 = ma*np.sum((va-v[0])**2)+mb*np.sum((vb-v[0])**2)
    ki1 = ma*np.sum((va+dv-a.velocities[0])**2)+mb*np.sum((vb+dv-a.velocities[0])**2)
    assert abs(ki1-ki0) < 1e-12

def test_infeasible_and_constrained_requests():
    x, v, m, t = sample(); a = retarget(v, m, x, t, float(np.sum(t*v)))
    with pytest.raises(ValueError, match='attainable'):
        retarget(v, m, x, t, a.attainable_max+1)
    with pytest.raises(ValueError, match='constrained'):
        retarget(v, m, x, np.zeros_like(v), 0)
    with pytest.raises(ValueError, match='positive'):
        retarget(v, -m, x, t, 0)
    with pytest.raises(ValueError, match='shape'):
        retarget(v, m, x, t, 0, np.ones_like(v))

def test_nearest_solution_against_independent_constrained_optimizer():
    from scipy.optimize import minimize
    x, v, m, t = sample(); wanted = float(np.sum(t*v))+.5
    out = retarget(v, m, x, t, wanted)
    p0 = np.sum(m[:, None]*v, axis=0)
    l0 = np.sum(np.cross(x, m[:, None]*v), axis=0)
    k0 = np.sum(m[:, None]*v*v)
    def constraints(flat):
        w = flat.reshape(v.shape)
        return np.r_[np.sum(m[:, None]*w, axis=0)-p0,
                     np.sum(np.cross(x, m[:, None]*w), axis=0)-l0,
                     np.sum(m[:, None]*w*w)-k0, np.sum(t*w)-wanted]
    def cost(flat):
        return .5*np.sum(m[:, None]*(flat.reshape(v.shape)-v)**2)
    fit = minimize(cost, v.ravel(), method='SLSQP',
                   constraints={'type':'eq','fun':constraints},
                   options={'ftol':1e-11,'maxiter':200})
    assert fit.success, fit.message
    assert np.max(np.abs(constraints(fit.x))) < 1e-8
    assert abs(cost(fit.x)-cost(out.velocities.ravel())) < 1e-8

def test_affine_relative_speed_reversal_is_involution():
    x, v, m, t = sample(); offset = .73
    old = float(np.sum(t*v)); wanted = 2*offset-old
    a = retarget(v, m, x, t, wanted)
    assert abs((a.raw_after-offset)+(old-offset)) < 1e-12
    b = retarget(a.velocities, m, x, t, 2*offset-a.raw_after)
    np.testing.assert_allclose(b.velocities, v, atol=2e-12)
