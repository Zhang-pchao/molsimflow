# Molecular reconstruction and staged contact observations

The public API is in molsimflow.postprocess.surface_reconstruction. It reduces
already identity-aligned molecular observations. It does not read trajectories,
assign chemistry, construct gas graphs, calculate accessible volume, fit models,
or infer whole-bubble motion.

## Frozen molecular fate

reconstruction_outcome(origin_water, onset_water, current_water, origin_gas,
current_gas, lost_cells, *, chemistry_valid=True) returns the inherited
molecule-supported fate and identity-flux dictionary without changing its
thresholds. Water records contain ids and observed Boolean normal flags.
Gas records contain ids, bins, components and main. Additional keys
are allowed, but arrays used by the reducer must align with unique identities.

The origin-to-onset loss must be at least three oxygen identities by net count
and three normal waters by net count. Oxygen and normal-water recovery bands
are deficit (<= 0.2), partial (> 0.2 and < 0.8), and refill (>= 0.8).
Discordant bands produce CHEMISTRY_DISCORDANT_COUNTS.

A gas-supported outcome requires all of the following:

- At least two net additional contact molecules in the fixed region.
- At least two net additional contact molecules in the lost columns.
- An intersecting contact component containing at least three molecules.
- At least two new contact identities, absent from the origin contact set,
  inside the lost columns and individually belonging to components of size >= 3.

Preexisting gas, isolated newcomers, out-of-column gain and constant-count
identity turnover cannot create replacement. Gross identity fluxes are kept
separate from net stoichiometry. The count-only diagnostic label is retained
but cannot override the molecular label. The main flag annotates membership
in the all-gas graph. A main-only result requires a separate call with its
independently computed graph; filtering all-gas component sizes is invalid.

Missing requested chemistry must be passed as chemistry_valid=False.
The returned record then contains only molecular_fate="UNOBSERVED" and
observation_status="CHEMISTRY_UNQUALIFIED". Unknown Boolean values such as
NaN are rejected rather than cast to true.

## Inclusive sampled confirmation

confirm_reconstruction(labels, geometry_ok, times_ps, epochs, endpoint, *,
duration_ps, expected_dt_ps, stride=1, source_ids=None, chemistry_valid=None)
returns a confirmation status. Supply actual observed times, source/epoch
identity and cadence; no step-to-time or molecule-count constants exist.

Both dwell endpoints must be present: 20 ps at 0.1 ps requires 201 frames.
Duration must contain a whole number of cadence intervals. A source change,
epoch seam, cadence gap, missing chemistry, inadequate geometry, or short
observation window cannot become stable confirmation. An observed change of
occupancy is COMPLETE_CHANGED; it is separate from censoring.

String "UNOBSERVED", None and integer -1 are unknown labels. Other integer
label schemes must provide an explicit chemistry-valid mask. Stride changes
only label sampling for a declared cadence sensitivity. All native chemistry,
geometry, and cadence observations remain mandatory. Supply the upstream
cumulative geometry qualification, and a cumulative chemistry-history mask
when the event contract requires valid prehistory. This reducer must not
replace cumulative accessible-capacity accounting with instantaneous volume.

Stable occupancy permits identity exchange. Confirmation of local gas
occupancy establishes neither whole-bubble attachment nor net translation.

## Connected-column anchored departure

same_column_anchor_departure(before_water, after_water, shape, *,
connectivity=8) additionally requires bins and observed anchored flags.
Anchored flags must be a subset of normal-water flags. It reuses the existing
periodic grid-component routine.

The dominant positive-oxygen-loss component is chosen using inherited summed
loss and grid-order tie handling. Its gate requires:

- Initial normal-water count >= 5, net oxygen loss >= max(3, ceil(0.3 * initial
  oxygen count)), and net normal-water loss >= 3 in the observed fixed region.
- At least two connected lost cells, with net oxygen loss >= 3 and net
  normal-water loss >= 3 in those same cells.
- At least three originally anchored identities from those cells absent from
  the entire observed fixed region at onset, and net anchored-water loss >= 1.

Departures in another component, conversion of retained oxygen identities,
within-region movement and equal-count molecular turnover cannot satisfy the
strong gate. The returned component, local counts and anchored departures
remain inspectable even when the final gate fails. This is a stronger
qualification; it does not relabel historical outcomes.

## Supply, arrival, establishment and maintenance

transition_ledger(supply, arrival, contact, times_ps, epochs, *,
expected_dt_ps, lag_ps, dwell_ps, persistence_ps, source_ids=None,
observation_valid=None) composes the existing sampled_episodes and
establishment_outcome helpers. States are 1, 0 and -1 for observed presence,
observed absence and unknown. Times must be strictly increasing physical
timestamps; reset timestep counters must be mapped before calling.

The adapter must define arrival independently of supply and supported contact.
The ledger does not infer arrival from successful contact. It records each
stage's episodes, all supply-episode origins and their left-censoring, explicit
arrival observations, establishment by lag, and subsequent contact maintenance.
Origins are selected from supply observations without future-outcome selection.
Origins with existing contact remain explicitly NOT_AT_RISK.

Establishment is dated at its confirmation, never backdated to first contact.
Unknown contact interrupts a pending dwell. A later complete dwell can still
confirm, while unresolved earlier observations keep first passage unknown.
Seams and source gaps cannot be crossed. No-supply samples, unknown-supply
samples, supplied origins and supplied contact-free origins have separate
denominators; repeated origins are correlated observations, not independent
replicates.

## Verification scope

Focused tests include a complete fixed fate/flux dictionary, strict positive
gas/refill controls, false replacement controls, identity exchange, separate
gas graphs, connected-column departures, conversion, unknown chemistry,
source/epoch seams, inclusive dwells and distinct transition failures.
Synthetic positive paths are software verification, not observed physical
events. Real-cache parity, source contracts, parent/site qualification and
scientific acceptance belong in the project adapter's versioned records.
