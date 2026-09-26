# TPCL observable qualification

`molsimflow.postprocess.tpcl_observable_qualification` is the Stage A gate before
any new pulse/reversal molecular-dynamics matrix is frozen.

## Input scope

The command consumes the immutable six-run high-frequency force-step package
and the accepted v3 kinematic tables. It uses the existing 10 fs coordinate
phase through 20 ps; it does not require new MD.

## Observable separation

For leading and trailing axis-tail regions in x and y, the analysis separates:

1. persistent-water displacement relative to the top substrate;
2. water entry into and exit from the region;
3. H-bond changes among water molecules that remain in the region;
4. apparent H-bond changes caused by region membership changes; and
5. continuous water-water and surface-water pair lifetimes.

The event detector is challenged with coherent, one-edge, opposing-edge, and
retreat injections over a fixed amplitude/duration grid. Injections are added
to 2 ps-detrended real F0 edge noise. The resulting table is an operating
envelope, not a proof that unobserved physical events do not exist.

Global water momentum and work/heat accounting are also evaluated. These
quantities do not define a unique atom-wise or hydrogen-bond-wise dissipation
partition for a many-body potential.

## Invocation

```bash
PYTHONPATH=/path/to/code_snapshot \
python -m molsimflow.postprocess.tpcl_observable_qualification \
  --package-root /path/to/four_interface_tpcl_high_frequency_force_on_v1_r3 \
  --reference-analysis /path/to/high_frequency_tpcl_mechanism_v3 \
  --output-dir /path/to/fresh/tpcl_observable_qualification_v1
```

The output directory is immutable and follows the numbered directory plan
written to `00_contract/DIRECTORY-PLAN.md`.

## Scientific boundary

A passing Stage A package means that the estimator domains, turnover
bookkeeping, and global conservation diagnostics were measured. It does not
establish water-network causality, a free-energy barrier, a converged event
rate, local dissipation, or a transferable friction law. Stage B remains a
separate review gate.
