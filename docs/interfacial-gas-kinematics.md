# Gas layer exchange and motion diagnostics

The interfacial_gas_kinematics module accepts identity-preserving molecular
coordinates and velocities. Callers supply pairing, PPP unwrapping and
largest-cluster membership. Layer labels are dynamic lower/middle/upper
quantiles, not a surface-contact definition. Lagged return can include
departure and re-entry; sampled survival excludes departures at sampled times
but cannot resolve events between frames. Cluster departures stay in the
denominator. Never concatenate reseeded or discontinuous identity epochs.

The least-squares velocity decomposition is translation plus omega cross r.
Positive omega_y corresponds to front-down/rear-up motion for migration along
+X above a lower horizontal solid. An instantaneous fit includes thermal
noise and does not establish rolling. Examine time blocks, cadence sensitivity,
coarse fields, and contact-population exchange. A projected XZ velocity field
averages over Y. It is not a central slice or proof of closed streamlines.

Simple shear provides an explicit counterexample: v_x=gamma*z, v_z=0 has
nonzero best-fit omega_y (gamma/2 for symmetric isotropic XZ support), while
material paths remain open with constant z. A nonzero fitted rotation or
vorticity therefore does not by itself identify rolling or a closed vortex.

The signed_planar_turning helper returns cadence-, epoch-, membership-, and radius-qualified sampled XZ angle increments. It is a trajectory diagnostic independent of the instantaneous rigid fit. Report rejected support and radius/cadence sensitivity. Opposite individual turns can cancel, and a shear path can turn without closing. Neither a net angle nor a full azimuth revolution alone proves coherent circulation, a closed three-dimensional orbit, or rolling.
