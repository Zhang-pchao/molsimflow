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
