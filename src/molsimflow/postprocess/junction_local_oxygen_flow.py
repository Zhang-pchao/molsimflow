"""PPP, face-owned original-solvent oxygen flow near a bubble footprint.

Original solvent oxygen is a Lagrangian population, not a dynamic H2O assignment.
No film, density, molecular temperature or detachment inference is returned.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from molsimflow.io.lammps_dump import minimum_image_vectors, periodic_center
from molsimflow.postprocess.nanobubble_local_flow import _region_stats


def oxygen_regions(oxygen_xyz: np.ndarray, oxygen_velocity: np.ndarray,
                   gas_centers: np.ndarray, site_xyz: np.ndarray,
                   site_groups: np.ndarray, bounds: np.ndarray, *,
                   edge_halfwidth_A: float = 5., height_A: float = 10.,
                   transverse_padding_A: float = 5., minimum_oxygen: int = 5) -> list[dict]:
    """Return front/rear/interior O velocities separately on both slab faces.

    Region edges use gas-center q05/q95 projections. Each oxygen belongs to its
    nearest original terminal site in full PPP; signed normal height excludes
    the slab interior. Material domains follow original site identities.
    """
    if min(edge_halfwidth_A,height_A,transverse_padding_A,minimum_oxygen)<=0:
        raise ValueError('Region widths and minimum count must be positive')
    oxygen_xyz=np.asarray(oxygen_xyz,dtype=float);oxygen_velocity=np.asarray(oxygen_velocity,dtype=float)
    gas_centers=np.asarray(gas_centers,dtype=float);site_xyz=np.asarray(site_xyz,dtype=float)
    site_groups=np.asarray(site_groups).astype(str);bounds=np.asarray(bounds,dtype=float)
    if oxygen_xyz.shape!=oxygen_velocity.shape or oxygen_xyz.ndim!=2 or oxygen_xyz.shape[1]!=3:
        raise ValueError('Oxygen positions and velocities require matching N by 3 arrays')
    if len(gas_centers)<2 or len(site_xyz)!=len(site_groups) or not len(site_xyz):
        raise ValueError('Nonempty surface and at least two gas centers required')
    if not all(np.isfinite(a).all() for a in (oxygen_xyz,oxygen_velocity,gas_centers,site_xyz,bounds)):
        raise ValueError('Nonfinite input')
    lengths=bounds[:,1]-bounds[:,0]
    if np.any(lengths<=0):raise ValueError('Invalid box')
    origin=bounds[:,0];center=periodic_center(gas_centers,bounds)
    local_gas=minimum_image_vectors(gas_centers-center,lengths)
    low,high=np.quantile(local_gas[:,:2],[.05,.95],axis=0)
    if np.any(high-low>=.8*lengths[:2]):raise ValueError('Broad periodic cluster requires topology review')
    local_o=minimum_image_vectors(oxygen_xyz-center,lengths)
    _,nearest=cKDTree((site_xyz-origin)%lengths,boxsize=lengths).query((oxygen_xyz-origin)%lengths)
    groups=site_groups[nearest];normal=minimum_image_vectors(oxygen_xyz-site_xyz[nearest],lengths)[:,2]
    types=np.full(len(oxygen_xyz),2,dtype=int);out=[]
    for face,sign in [('top',1),('bottom',-1)]:
        if not np.any(np.char.startswith(site_groups,face+'_')):raise ValueError('Missing slab face')
        h=normal*sign
        shared=np.char.startswith(groups,face+'_')&(h>=0)&(h<=height_A)&(local_o[:,1]>=low[1]-transverse_padding_A)&(local_o[:,1]<=high[1]+transverse_padding_A)
        for region,mask in [('front',np.abs(local_o[:,0]-high[0])<=edge_halfwidth_A),('rear',np.abs(local_o[:,0]-low[0])<=edge_halfwidth_A),('interior', (local_o[:,0]>low[0]+edge_halfwidth_A)&(local_o[:,0]<high[0]-edge_halfwidth_A))]:
            selected=shared&mask
            stats=_region_stats(selected,types,oxygen_velocity,h,{2:15.999},oxygen_type=2,minimum_oxygen=minimum_oxygen,volume_A3=0.,direction_index=0,transverse_index=1)
            row=dict(face=face,region=region,original_oxygen_count=int(selected.sum()),left_oxygen_count=int((selected&(groups==face+'_left')).sum()),right_oxygen_count=int((selected&(groups==face+'_right')).sum()),status='PASS' if selected.sum()>=minimum_oxygen else 'INSUFFICIENT_OCCUPANCY')
            for axis,key in [('x','water_drive_velocity_mps'),('y','water_transverse_velocity_mps'),('z','water_vertical_velocity_mps')]:
                value=float(stats[key]);row['oxygen_v'+axis+'_mps']=value if np.isfinite(value) else None
            out.append(row)
    return out
