# Two-anchor scaffold motif matching

`molsimflow.postprocess.scaffold_motif_match` supplies a small pure-array
descriptor for two anchor vectors relative to a fixed center. It does not load
coordinates, identify bonds, unwrap periodic boundaries, move terminal atoms,
assess sterics, or validate chemistry. The caller supplies an already eligible
source set and owns its provenance, motif membership and chemical signature.

## Declared mapping and score

For ordered anchor vectors v0 and v1, normalize them separately to u0 and u1.
The proper bridge basis has columns:

```text
x = normalize(u0 - u1)
z = -normalize(u0 + u1)
y = normalize(z cross x)
```

Roundoff-safe orthogonalization retains this chart. Zero-length, nonfinite,
collinear or numerically singular vectors are rejected. The default
collinearity guard is a dimensionless cross-product norm of 1e-12, not a
chemical angle-admission threshold.

`match_anchor_pair(source, target)` evaluates both target-anchor orders
(0,1) and (1,0). For each ordering:

```text
R = B_target @ B_source.T
transformed_source = source @ R.T
score_A = sqrt(mean_over_two_anchors(sum_over_xyz(residual_A ** 2)))
```

This uses the original anchor lengths; normalized vectors only define the
rotation. No arbitrary weighted mixture of angles and lengths is introduced.
The score is RMS vector residual per anchor, not RMS per Cartesian component.

The best score is the minimum of these two declared basis mappings. It is
not generally the minimum over every possible rotation (a Procrustes fit);
unequal anchor lengths can distinguish those estimands. The function makes
no displacement, strain-energy, chemical-quality or steric-admission claim.

## Orientation ambiguity

The result reports both scores, both proper rotations, both target orders,
the minimum score, `best_flip_indices`, and `ambiguous`. A best flip includes
any score within the declared arithmetic tolerance of the minimum.

Exchanging an anchor order permutes the two alternatives while preserving the
best score and the set of rotations. A common global proper rotation preserves
scores and transforms each mapping covariantly. Equal-length symmetric anchors
can have two distinct, equally scoring rotations. There is deliberately no
field claiming a unique chosen physical orientation in this case.

## Ranking and selecting one source

`rank_scaffold_motifs(source_ids, source_vectors, target_vectors,
preferred_source_id=None, tie_tolerance_A=1e-10)` returns exactly one selected
source ID and the full ranked evidence.

Source vectors have shape (N,2,3); the target has shape (2,3). Source IDs must be
unique integers. All vector units must be consistent; the API names use A.
Periodicity and relative-coordinate construction are external responsibilities.

At each rank level, form a group of remaining scores within 1e-10 A of the
remaining global minimum (or the explicitly supplied arithmetic tolerance).
Within that group prefer `preferred_source_id`, then the lowest source ID.
An absent preferred ID is ignored without error. A preferred ID outside the
minimum group cannot override a better group.

Using minimum-anchored groups avoids a nontransitive pairwise epsilon comparator:
scores at 0, 0.75t and 1.5t cannot all become tied through chaining. A selected
preferred ID may have a score slightly above the exact global minimum within
the tolerance, so both `selected_score_A` and `global_minimum_score_A` are
reported. The complete source records retain both orientation alternatives.

The caller must prefilter the correct chemical signature. The function does
not infer whether a requested self ID is chemically eligible. It only prefers
that ID if present in the supplied candidate set. Ranking is descriptive;
it does not authorize trying the next source after an external clash failure.

## Verification scope

The tests use invented vectors only. An independent oracle derives each score
from source/target anchor lengths and half the difference in included angles,
without constructing the bridge basis. Further tests cover global rotations,
anchor permutations, self matching, unequal lengths, proper determinants,
arithmetic ties, absent self IDs, nontransitive tie chains, invalid inputs and
immutable output records. No physical source motif or candidate is admitted
by these mathematical tests.
