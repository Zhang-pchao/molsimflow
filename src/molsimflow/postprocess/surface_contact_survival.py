"""All-origin sampled uninterrupted contact at registered material locations."""
from __future__ import annotations
import numpy as np
from molsimflow.postprocess.surface_contact_memory import region_recontact_curve


def sampled_region_survival(supported, times, epochs, lag_steps, *, frame_dt, valid=None):
    """Use exactly the endpoint recontact risk set, requiring every saved contact.

    This is an origin-weighted sampled overlap, not an event-duration distribution.
    No smoothing, gap bridging or independent-origin uncertainty is implied.
    """
    if not np.isfinite(frame_dt) or frame_dt <= 0:
        raise ValueError("A finite positive cadence is required")
    lags = np.asarray(lag_steps)
    endpoint = region_recontact_curve(supported, times, epochs, lags,
                                     frame_dt=frame_dt, valid=valid)
    s = np.asarray(supported)
    t = np.asarray(times, float)
    e = np.asarray(epochs)
    ok = np.ones(len(s), bool) if valid is None else np.asarray(valid, bool)
    joined = (e[1:] == e[:-1]) & np.isclose(np.diff(t), frame_dt, rtol=0, atol=1e-8)
    joined &= ok[1:] & ok[:-1]
    runs = np.r_[0, np.cumsum(~joined)]
    # Consecutive supported saved observations remaining from each frame.
    remaining = np.zeros(s.shape, dtype=np.int32)
    for i in range(len(s)-1, -1, -1):
        remaining[i] = s[i] & ok[i]
        if i+1 < len(s) and joined[i]:
            remaining[i] += np.where(s[i], remaining[i+1], 0)
    den = s.sum(axis=1)
    result = []
    for row in endpoint:
        lag = row['lag_steps']
        a = np.arange(max(0, len(s)-lag))
        b = a+lag
        use = ok[a] & ok[b] & (runs[a] == runs[b]) & (den[a] > 0)
        a = a[use]
        assert len(a) == row['n_origins']
        values = (remaining[a] > lag).sum(axis=1)/den[a]
        survival = float(values.mean()) if len(a) else None
        again = row['mean_original_region_fraction']
        if survival is not None:
            assert -1e-12 <= survival <= again+1e-12
        result.append(dict(row, sampled_uninterrupted_fraction=survival,
                           interrupted_then_recontacted_fraction=None if survival is None else max(0., again-survival)))
    return result
