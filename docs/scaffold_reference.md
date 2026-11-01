# Immutable common-scaffold reference

`molsimflow.postprocess.scaffold_reference` freezes an explicitly selected Si /
bridge-O scaffold and registers frames by rigid translation of those same atom
IDs. Terminal OH/CH3 atoms never enter this reference. The module consumes
already-read arrays and reuses `io.lammps_dump.minimum_image_vectors`; it adds no
trajectory reader, chemical assignment, rotational fit or simulation operation.

The existing `surface_reference.SurfaceReference` has a different contract: it
tracks a supplied slab center using periodic Z. It remains unchanged.

## API and units

All coordinates, tolerances and shifts below are in angstroms. IDs are positive
integers; element labels are case-sensitive symbols (`Si`, `O`).

```python
reference = build_scaffold_reference(
    atom_ids, elements, coordinates_A, bounds_A,
    silicon_ids=unchanged_si_ids,
    bridge_oxygen_ids=unchanged_bridge_O_ids,
    top_ids=explicit_top_scaffold_ids,
    deformation_tolerance_A=qualified_maximum_residual,
    wrapping_tolerance_A=1e-8,
    periodic=(True, True, False),
)
registration = register_scaffold(
    reference, frame_atom_ids, frame_elements, frame_xyz_A, frame_bounds_A,
    periodic=(True, True, False),
)
aligned_xyz_A = register_coordinates(frame_xyz_A, reference, registration)
frame_site_xy_A = reference_to_frame_sites(fixed_site_xy_A, reference, registration)
```

The Si list must be nonempty; the bridge-O list may be empty for an explicitly
Si-only scaffold. The lists must be disjoint. Every selected Si ID must have
symbol `Si`, and every selected bridge ID must have symbol `O`. This does not
prove that O is a bridge: its unchanged bonding/chemical eligibility requires
external evidence. `top_ids` must be a nonempty subset of the selected scaffold.
No terminal-height guess or automatic top-atom selection occurs here.

`ScaffoldReference` and `ScaffoldRegistration` are frozen records containing
immutable scalar/tuple data, with defensive conversion even when constructed or
replaced directly with NumPy arrays. Mutating caller arrays cannot alter them.
Reference properties include `sha256` and `height_datum_A`; registration exposes
`translation_A`, `max_residual_A`, and `reference_hash`.

## Registration and fixed sites

Only axis-aligned `(3, 2)` finite bounds with positive lengths and explicitly
boolean p-p-f periodicity are accepted. The frame's full bounds must exactly
match the reference bounds, including origin. Changed cells, moved box origins,
triclinic boxes and periodic Z require a separately designed observer.

Atom matching uses IDs, independent of row order. Missing or duplicated IDs and
changed scaffold elements fail. Displacements are frame minus reference. For XY,
minimum-image displacements are unwrapped around the smallest-ID scaffold atom
before taking their componentwise median. Z uses an ordinary median with no
wrapping. Every selected atom's residual Euclidean norm must be at most the
frozen deformation tolerance (with a 1e-12 numerical comparison allowance).
Half-box consensus shifts and half-box differences from the anchor are rejected
within the declared wrapping tolerance because the image choice is ambiguous.
Both tolerances must be below one quarter of the shortest periodic box length;
the deformation tolerance may be zero and the wrapping tolerance must be positive.

`register_coordinates` subtracts the fitted XYZ shift, then wraps XY into the
reference box. `reference_to_frame_sites` adds the fitted XY shift to fixed
reference-frame site centers and wraps XY into the same box. Thus fixed sites and
observations can be compared in either consistent coordinate frame across a PBC
crossing. Both helpers reject a registration belonging to a different reference.
The XY shift is defined modulo whole-box translations; it is not a cumulative
trajectory displacement and cannot by itself establish net physical movement.

## Fixed datum versus physical observation layer

`height_datum_A` is the median **reference** z of `top_ids`. It stays constant
across arms and frames. Rigid Z motion is removed by registration, so a caller's
physical relative-height convention is explicitly

```text
height = z_frame - registration.translation_A[2]
         - (reference.height_datum_A + separately_frozen_layer_offset_A)
```

The module does not infer that offset or silently replace an established 0-3 A
water layer. A source-only, outcome-independent mapping from a qualified earlier
height convention may be documented in a project adapter. A local roughness grid
is also a separate observer product: this median datum is not a height map.
Changing terminal atoms must not rebuild the datum, registration scaffold or
fixed-site reference in each treatment arm.

## Reference identity and limits

The SHA256 payload includes sorted selected atom IDs, corresponding elements and
reference coordinates, explicit Si/bridge/top selections, full bounds, p-p-f
periodicity, registration/height conventions and both tolerances. XY reference
coordinates are wrapped to the box. Floats use deterministic hexadecimal
serialization with normalized signed zero. Hashes use exact represented values,
not approximate geometry matching or numeric rounding; upstream references that
differ only through floating-point serialization still need explicit provenance
comparison. Terminal atoms are excluded, so their element or position changes do
not alter the reference hash. The same reference object/hash should be supplied
to both controlled treatment arms.

This is a conservative rigid-scaffold observer. It does not establish scaffold
chemical stability, PES validity, dynamic suitability of a tolerance, or the
physical validity of a layer offset. Rotation, shear, flexibility or deformation
above the frozen tolerance fails instead of being absorbed into a moving
observer. Thermal trajectories require a separately qualified tolerance and
residual audit; a successful static identity test does not qualify dynamics.
A selected scaffold can only constrain its own atoms, not unobserved deformation
elsewhere in a surface.

## Verification

```bash
python -m pytest -q -p no:cacheprovider tests/test_scaffold_reference.py
```

Tests cover input-permutation hashes, immutable records, terminal-only changes,
matched-ID registration, XY PBC crossings, rigid Z translation, fixed sites,
explicit layer offsets, missing/duplicate/changed IDs or elements, deformation,
ambiguous wrapping, non-p-p-f/changed/triclinic boxes, observer-hash binding and
invalid reference selections. These synthetic tests establish software behavior,
not physical validation of a new intervention or a production trajectory.
