# Local azimuth callback

One MPI rank, fixed atom IDs, orthorhombic PPP cells. Compile with the matching installed LAMMPS/MPI libraries. Create `fix CV all external pf/callback 1 1` with `suffix off`, then restore the desired engine suffix. Register the callback before the first run. Enable `fix_modify CV energy yes virial yes`.

Configuration: atom_count H_ID O_ID S_ID R_ID kappa reference_radians force_limit H_type O_type C_type. Every field is mandatory. Atom order for the reference function is H,O,S,R. The reference types are caller-specific; coordinates and IDs must remain in the admitted local geometry and fixed-topology domain. The callback aborts on competing O/C within 1.3 length units of H, another H within 1.3 of O, OH outside 0.7–1.3, either S–O edge outside 1.2–2.2, or either transverse projection <=0.2. These example guard lengths assume angstrom units and require project-specific review.

Output vector: cos(phi-reference), bias energy, H bias force norm, maximum bias force norm among O/S/R. Engine PE already contains global bias energy; do not add it again. All four reaction forces and the local many-body virial are included. This is a conservative diagnostic restraint, not a physical model or a kinetic estimator.
