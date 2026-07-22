# Incremental `md_postprocess` Audit — 2026-07-22

## Scope

This review compares the existing migration baseline with the source
`md_postprocess` tree after the 2026-06-20 incremental audit.  The purpose is to
identify genuinely new reusable analysis kernels, not to copy every changed
plot or wrapper.

## Result

Only one non-generated Python file was newer than the audit cutoff:

| Source file | Classification | Decision |
|---|---|---|
| `scripts/render_toc_concept_graphic.py` | Publication/TOC graphic generator with fixed project artwork, labels, and output assumptions | Excluded from the public package |

No new reusable trajectory reader, descriptor calculation, FES kernel,
topology summary, or generic plotting primitive was found after the cutoff.
The existing migration modules therefore remain the correct implementation
baseline for this review period.

## Engineering Follow-up

The in-progress spherical-interface analysis was retained as a reusable
workflow and hardened before publication:

- atom-type IDs are explicit inputs rather than embedded constants;
- case ordering, termination fractions, and plot colors are derived from the
  supplied cases rather than private case names;
- the trajectory filename is generic by default and the data filename is
  optionally explicit;
- the plotting dependency is declared in the `analysis` installation extra;
- the public-repository audit now checks SSH aliases, IPv4 addresses,
  token-like strings, private-key markers, and credential assignments.

Project-specific graphics, scheduler wrappers, real simulation outputs, and
private run metadata remain outside the tracked package.
