"""Exact discrete water-column bookkeeping with explicit geometry changes."""
import numpy as np

def column_membership(xyz,qualified,top,gap,supported,lengths,shape):
    """Return fixed XY cell index or -1 outside a qualified observed gap."""
    xyz=np.asarray(xyz,float);qualified=np.asarray(qualified,bool)
    lengths=np.asarray(lengths,float);shape=np.asarray(shape,int)
    ij=np.floor((xyz[:,:2]%lengths[:2])/lengths[:2]*shape).astype(int)
    bins=ij[:,0]*shape[1]+ij[:,1]
    z=xyz[:,2]-top[bins]
    inside=qualified&supported[bins]&(z>.5)&(z<gap[bins]-.5)
    return np.where(inside,bins,-1).astype(np.int16)

def ordered_column_balance(previous_xyz,current_xyz,previous_water,current_water,
                           previous_top,current_top,previous_gap,current_gap,
                           previous_supported,current_supported,lengths,shape):
    """Telescoping lateral -> vertical -> geometry -> chemistry accounting.

    The ordering is a bookkeeping convention, not a causal decomposition.
    Only columns supported at both endpoints are qualified. Changes involving
    an unobserved envelope are marked separately by the common-support mask.
    """
    common=np.asarray(previous_supported,bool)&np.asarray(current_supported,bool)
    xy=np.asarray(previous_xyz,float).copy();xy[:,:2]=np.asarray(current_xyz)[:,:2]
    args=[(previous_xyz,previous_water,previous_top,previous_gap),
          (xy,previous_water,previous_top,previous_gap),
          (current_xyz,previous_water,previous_top,previous_gap),
          (current_xyz,previous_water,current_top,current_gap),
          (current_xyz,current_water,current_top,current_gap)]
    memberships=[column_membership(x,q,t,g,common,lengths,shape) for x,q,t,g in args]
    n=int(np.prod(shape))
    counts=np.stack([np.bincount(b[b>=0],minlength=n) for b in memberships])
    delta=np.diff(counts,axis=0)
    np.testing.assert_array_equal(delta.sum(axis=0),counts[-1]-counts[0])
    return common,memberships,delta
