"""Diagnose first saved-frame local support loss without calling it evacuation."""
from __future__ import annotations
import numpy as np
from molsimflow.postprocess.surface_loss_census import prospective_count_losses


def diagnose_region_support_losses(supported, near_counts, shell_counts,
                                   main_shell_counts, times, epochs, *, frame_dt,
                                   minimum_support=3):
    """Return every adjacent true-to-false transition with explicit count classes.

    Arrays have shape (frames, regions), all entries must be qualified; unknown
    geometry must not be encoded as zero. Reuses the source-aware loss census.
    Classes describe the first failed saved frame, not a continuous-time path.
    """
    s=np.asarray(supported)
    arrays=[np.asarray(x) for x in [near_counts,shell_counts,main_shell_counts]]
    if s.ndim!=2 or s.dtype!=np.bool_ or any(x.shape!=s.shape for x in arrays):
        raise ValueError('Boolean support and aligned count matrices required')
    if any(not np.isfinite(x).all() or np.any(x<0) or np.any(x!=np.floor(x)) for x in arrays):
        raise ValueError('Unknown geometry cannot be classified as no gas')
    n,c,m=arrays
    if minimum_support<1 or np.any(m>c) or np.any(c>n) or np.any(s & (m<minimum_support)):
        raise ValueError('Inconsistent nested counts or support')
    census=prospective_count_losses(s.astype(np.int8),np.asarray(times),np.asarray(epochs),
        expected_dt=frame_dt,history_span=0,search_span=frame_dt,minimum_initial_count=1,
        minimum_loss=1,fractional_loss=0,history_absolute_tolerance=0,history_fractional_tolerance=0)
    starts,regions=np.nonzero(census.first_loss_index>=0)
    result=[]
    for a,p in zip(starts,regions):
        b=int(census.first_loss_index[a,p])
        label=('EMPTY_NEAR_REGION' if n[b,p]==0 else
               'NEAR_GAS_NO_SHELL_CONTACT' if c[b,p]==0 else
               'SHELL_CONTACT_OUTSIDE_MAIN' if m[b,p]==0 else
               'MAIN_CONTACT_BELOW_SUPPORT' if m[b,p]<minimum_support else
               'MAIN_CONTACT_DISCONNECTED')
        result.append(dict(start_index=int(a),end_index=b,region_index=int(p),
            near_before=int(n[a,p]),near_after=int(n[b,p]),
            shell_before=int(c[a,p]),shell_after=int(c[b,p]),
            main_shell_before=int(m[a,p]),main_shell_after=int(m[b,p]),
            near_gas_present=bool(n[b,p]>0),diagnostic=label))
    return result
