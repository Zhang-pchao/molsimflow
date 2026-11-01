"""Reaction-aware oxygen occupation in a fixed registered surface cylinder."""
import numpy as np

def oxygen_patch_members(xyz, anchor_xy, top, lengths_xy, radius=4., height=3.):
    xyz=np.asarray(xyz,float);anchor=np.asarray(anchor_xy,float);L=np.asarray(lengths_xy,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or anchor.shape!=(2,) or L.shape!=(2,) or np.any(L<=0):
        raise ValueError("Expected finite oxygen coordinates and positive XY box")
    if not np.isfinite(xyz).all() or not np.isfinite(anchor).all() or not np.isfinite(top):
        raise ValueError("Missing geometry cannot mean empty water")
    delta=xyz[:,:2]-anchor;delta-=np.rint(delta/L)*L
    return np.flatnonzero(((delta*delta).sum(axis=1)<=radius*radius)&(xyz[:,2]>=top)&(xyz[:,2]<top+height))

def occupation_change(ids0,category0,ids1,category1):
    """Separate actual O exchange from H2-category changes of retained oxygen."""
    a=dict(zip(map(int,ids0),map(int,category0)));b=dict(zip(map(int,ids1),map(int,category1)))
    if len(a)!=len(ids0) or len(b)!=len(ids1) or len(ids0)!=len(category0) or len(ids1)!=len(category1):
        raise ValueError("Unique matching oxygen identities required")
    left=a.keys()-b.keys();entered=b.keys()-a.keys();retained=a.keys()&b.keys()
    spatial=sum(b[k]==2 for k in entered)-sum(a[k]==2 for k in left)
    chemical=sum(int(b[k]==2)-int(a[k]==2) for k in retained)
    dh2=sum(v==2 for v in b.values())-sum(v==2 for v in a.values())
    assert dh2==spatial+chemical
    return dict(delta_O=len(b)-len(a),O_left=len(left),O_entered=len(entered),
                delta_H2=dh2,H2_spatial_change=spatial,H2_retained_category_change=chemical)
