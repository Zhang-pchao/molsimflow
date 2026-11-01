"""Frozen-caliper event matching without outcome-dependent selection."""
import numpy as np
from scipy.spatial import cKDTree
G_FIELDS=('gap_median_A','gap_q25_A','gap_q75_A','volume_A3','approach_A_ps',
          'accessible_N2','main_N2','seed_N2','front_distance_A','boundary_A',
          'curvature_A2','roughness_A','supported_column_fraction')
C_FIELDS=('OH_count','CH3_count','central_type','other_H0','other_H1','other_H3plus','other_ambiguous')
ABS_CALIPERS={'gap_median_A':.5,'approach_A_ps':.02,'accessible_N2':2.,
              'seed_N2':3.,'front_distance_A':1.,'roughness_A':.25}

def discovery_scales(rows):
    """Fix geometry SDs using discovery covariates only."""
    values=np.array([[float(r[k]) for k in G_FIELDS] for r in rows],float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Finite discovery covariates required')
    return dict(zip(G_FIELDS,np.std(values,axis=0,ddof=1 if len(values)>1 else 0)))

def greedy_caliper_pairs(rows,scales,*,composition=True,groups=None):
    """Disjoint deterministic closest pairs; groups optionally enforce high/low.

    No outcome is read. Categorical branch/face matching is always mandatory.
    Returned indices refer to the input rows. A zero discovery SD requires exact
    equality. Calipers are never widened to improve coverage.
    """
    if not rows:return []
    matrix=np.array([[float(r[k]) for k in G_FIELDS] for r in rows],float)
    if not np.isfinite(matrix).all():raise ValueError('Finite baseline required')
    sd=np.array([scales[k] for k in G_FIELDS]);vary=sd>0
    norm=matrix[:,vary]/sd[vary] if vary.any() else np.zeros((len(rows),1))
    candidates=cKDTree(norm).query_pairs(.25,p=np.inf,output_type='ndarray')
    cat=('branch','face')+(C_FIELDS if composition else ())
    accepted=[]
    for i,j in candidates:
        if groups is not None and (groups[i] is None or groups[j] is None or groups[i]==groups[j]):continue
        if any(str(rows[i][k])!=str(rows[j][k]) for k in cat):continue
        d=np.abs(matrix[i]-matrix[j])
        if np.any(d[~vary]>1e-12):continue
        if any(abs(float(rows[i][k])-float(rows[j][k]))>v for k,v in ABS_CALIPERS.items()):continue
        b0,b1=float(rows[i]['boundary_A']),float(rows[j]['boundary_A'])
        if abs(b0-b1)>.2*max(abs(b0),abs(b1)):continue
        accepted.append((float(np.sum(d[vary]/sd[vary])),int(i),int(j)))
    used=set();pairs=[]
    for distance,i,j in sorted(accepted):
        if i in used or j in used:continue
        used.update((i,j));pairs.append((i,j,distance))
    return pairs

def frozen_ecdf(training,values):
    training=np.sort(np.asarray(training,float));values=np.asarray(values,float)
    if not len(training) or not np.isfinite(training).all():raise ValueError('Finite nonempty discovery reference required')
    return np.searchsorted(training,values,side='right')/len(training)
