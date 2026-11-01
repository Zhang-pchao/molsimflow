"""Moment-based mean shape and descriptive overdamped relaxation."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

def moment_axes(points: np.ndarray) -> tuple[float,float,float]:
    """Equivalent uniform ellipsoid axes; points must be unwrapped."""
    delta=np.asarray(points,float)-np.mean(points,axis=0)
    a=float(np.sqrt(2.5*np.mean(np.sum(delta[:,:2]**2,axis=1))))
    c=float(np.sqrt(5*np.mean(delta[:,2]**2)))
    return a,c,float(4*np.pi*a*a*c/3)

def constant_volume_radius(c, a0: float, c0: float):
    """Conditional geometry relation, not a prediction of c(t)."""
    c=np.asarray(c,float)
    if np.any(c<=0) or a0<=0 or c0<=0:raise ValueError("Axes must be positive")
    return a0*np.sqrt(c0/c)

def relaxation_curve(t,Hinf,H0,tau):
    return Hinf+(H0-Hinf)*np.exp(-np.asarray(t)/tau)

def fit_relaxation(t,y):
    """Three-parameter descriptive fit with declared numerical bounds."""
    t=np.asarray(t,float);y=np.asarray(y,float)
    if len(t)<6 or np.any(np.diff(t)<=0):raise ValueError("Six ordered observations required")
    x=t-t[0];duration=x[-1];lower=max(np.min(np.diff(t))/2,1e-6);upper=20*duration
    bounds=([0,0,lower],[100,100,upper])
    fits=[]
    for tau in [duration/10,duration/2,duration,5*duration]:
        fits.append(least_squares(lambda p:relaxation_curve(x,*p)-y,[np.clip(y[-1],.01,99.99),np.clip(y[0],.01,99.99),tau],bounds=bounds,max_nfev=4000))
    r=min(fits,key=lambda z:np.sum(z.fun*z.fun));pred=relaxation_curve(x,*r.x)
    ss=float(np.sum((y-y.mean())**2))
    return dict(Hinf=float(r.x[0]),H0=float(r.x[1]),tau_ps=float(r.x[2]),t0_ps=float(t[0]),rmse=float(np.sqrt(np.mean((pred-y)**2))),R2=float(1-np.sum((pred-y)**2)/ss) if ss>0 else None,at_bound=bool(np.any(np.isclose(r.x,bounds[0],atol=.01)) or np.any(np.isclose(r.x,bounds[1],rtol=.001))),tau_bounds_ps=[lower,upper],prediction=pred.tolist())


def cap_from_area_volume(area: float, volume: float) -> dict:
    """Uniform spherical-cap reference, including molecular-center quantiles.

    Area and volume are inputs measured without using the observed height.
    The cap is a reference manifold; actual shapes need not be spherical caps.
    """
    from scipy.optimize import brentq
    if not np.isfinite([area, volume]).all() or area <= 0 or volume <= 0:
        raise ValueError("Positive finite footprint and volume required")
    a2 = area / np.pi
    h = brentq(lambda h: np.pi*h*(3*a2+h*h)/6-volume,
               0.0, (6*volume/np.pi)**(1/3)*1.01)
    zc = (h*h-a2)/(2*h)
    def cumulative(z):
        return np.pi*(a2*z+zc*z*z-z*z*z/3)
    quantiles = [brentq(lambda z: cumulative(z)-p*volume, 0, h)
                 for p in [.05, .5, .95]]
    return dict(cap_height=float(h), radius=float(np.sqrt(a2)),
                q05=quantiles[0], q50=quantiles[1], q95=quantiles[2],
                quantile_height=float(quantiles[2]-quantiles[0]))


def lower_profile_footprint(z, areas, *, plane=0.0, band=(3.0, 8.0)):
    """Extrapolate a footprint from only the lower cross-sectional profile.

    For a cap, A(z)/pi + z**2 = a**2 + 2*zc*z. This observer uses
    no top position, global fitted height, or volume. It is a geometric
    extrapolation and must not be called dry solid-gas contact area.
    """
    z=np.asarray(z,float)-plane; areas=np.asarray(areas,float)
    mask=(z>=band[0])&(z<=band[1])&(areas>0)&np.isfinite(areas)
    if mask.sum()<4 or np.ptp(z[mask])<2:
        return dict(available=False, reason="INSUFFICIENT_LOWER_PROFILE")
    target=areas[mask]/np.pi+z[mask]**2
    design=np.c_[np.ones(mask.sum()),z[mask]]
    intercept,slope=np.linalg.lstsq(design,target,rcond=None)[0]
    if intercept<=0:
        return dict(available=False, reason="NONPOSITIVE_EXTRAPOLATED_AREA")
    prediction=np.pi*(intercept+slope*z[mask]-z[mask]**2)
    return dict(available=True, area=float(np.pi*intercept),
                profile_rmse=float(np.sqrt(np.mean((prediction-areas[mask])**2))),
                profile_relative_rmse=float(np.sqrt(np.mean((prediction-areas[mask])**2))/areas[mask].mean()),
                lower_profile_zc=float(slope/2), points=int(mask.sum()))


def density_component_profile(density, z, spacing, threshold):
    """Largest connected 3-D isodensity component and its area profile."""
    from scipy.ndimage import label
    d=np.asarray(density,float)
    labels,n=label(d>=threshold,structure=np.ones((3,3,3),int))
    if n==0: raise ValueError("No isodensity component")
    mass=np.bincount(labels.ravel(),weights=d.ravel()); mass[0]=0
    mask=labels==int(mass.argmax())
    area=mask.sum(axis=(0,1))*float(spacing[0]*spacing[1])
    volume=float(area.sum()*spacing[2])
    edge=bool(mask[[0,-1],:,:].any() or mask[:,[0,-1],:].any())
    return dict(areas=area, volume=volume,
                density_mass_fraction=float(d[mask].sum()/d.sum()),
                component_fraction=float(mask.sum()/(d>=threshold).sum()),
                boundary_touch=edge, components=int(n))


def driven_shape_response(time, driver, tau: float, initial: float):
    """Exact causal integration for a piecewise-linear measured input.

    Values through t[i] depend only on inputs through t[i]. tau=0 is the
    instantaneous model, including its first output value.
    """
    t=np.asarray(time,float); g=np.asarray(driver,float)
    if t.ndim!=1 or g.shape!=t.shape or len(t)==0 or not np.isfinite(t).all() or not np.isfinite(g).all() or np.any(np.diff(t)<=0):
        raise ValueError("Finite ordered input vectors required")
    if not np.isfinite(tau) or tau<0: raise ValueError("Nonnegative tau required")
    if tau==0:return g.copy()
    y=np.empty_like(g);y[0]=initial
    for i,dt in enumerate(np.diff(t),1):
        one_minus=-np.expm1(-dt/tau)
        ramp=1-(tau/dt)*one_minus
        y[i]=(1-one_minus)*y[i-1]+one_minus*g[i-1]+ramp*(g[i]-g[i-1])
    return y


def profile_shape_response(time, driver, observed, train, tau_values, *,
                           additive_offset=False, offset_basis=None):
    """Fit only on the declared prefix; retain all training-loss profiles.

    An optional offset is the same nuisance degree of freedom in static and
    dynamic models. It diagnoses observation bias, not a capillary parameter.
    offset_basis optionally prescribes its time-dependent forcing shape; its
    coefficient is still fitted exclusively on calibration observations.
    """
    t=np.asarray(time,float);g=np.asarray(driver,float);h=np.asarray(observed,float)
    train=np.asarray(train,bool).copy()
    if len(t)!=len(h) or train.shape!=t.shape or not train[0]:
        raise ValueError("Initial observed state must be in calibration prefix")
    train[0]=False
    if train.sum()<4:raise ValueError("Four noninitial calibration bins required")
    correction=np.ones_like(g) if offset_basis is None else np.asarray(offset_basis,float)
    if correction.shape!=g.shape or not np.isfinite(correction).all():
        raise ValueError("Finite aligned correction basis required")
    if offset_basis is not None and not additive_offset:
        raise ValueError("An explicit correction basis requires additive_offset=True")
    profiles=[]
    for tau in tau_values:
        pred=driven_shape_response(t,g,float(tau),float(h[0]))
        basis=driven_shape_response(t,correction,float(tau),0.)
        offset=0.
        if additive_offset:
            if np.dot(basis[train],basis[train])<=0:
                raise ValueError("Correction basis must be nonzero in calibration")
            offset=float(np.dot(basis[train],h[train]-pred[train])/np.dot(basis[train],basis[train]))
            pred=pred+offset*basis
        mse=float(np.mean((pred[train]-h[train])**2))
        profiles.append(dict(tau=float(tau), offset=offset, train_mse=mse,
                             prediction=pred))
    best=min(profiles,key=lambda r:r['train_mse'])
    return best,profiles
