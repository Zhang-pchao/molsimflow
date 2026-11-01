# Fixed-region water identity exchange

`interfacial_contact_renewal.endpoint_identity_exchange` accepts boolean N by K
memberships for identical ordered particle IDs. It returns retained R, gained
G and lost L identities, with N0=R+L, N1=R+G, net=G-L and gross=G+L.
A constant count can coexist with complete replacement. Missing denominators
remain NaN. These are exact saved-endpoint set identities, not kinetic laws.

For an independently defined material region M0 at time 0, evaluate M0(X0) and
M0(X1) using the same geometry, local height reference and material labels.
Repeating with M1 held fixed provides an end-anchoring sensitivity. Evaluating
M0(X0) and M1(X1) instead measures membership in a moving region; do not call
that particle-only exchange. Region definitions must not be conditioned on
water scarcity, hydrogen bonds or favorable observed migration.

Summaries should report summed R/summed N0 with its support, rather than silently
averaging undefined ratios. Report lag, phase, source duration, material and
geometry sensitivity. Moving between region subdivisions counts as leaving a
subdivision; sum particle memberships before computing pooled exchange.
Weighted orientation channels are not literal identity sets.

Overlap at two endpoints includes unobserved exits and reentries. It is an
intermittent overlap, not continuous residence or a survival function. G and L
do not count all intermediate crossings and should not be divided by lag to
claim a converged flux. Exiting a finite height band is not thermodynamic
desorption. A high endpoint overlap alone does not establish an energetic
binding barrier or distinguish chemistry from shape or local confinement.
