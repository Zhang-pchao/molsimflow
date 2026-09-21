# High-frequency TPCL force-step analysis

`molsimflow.postprocess.tpcl_force_step_analysis` analyzes the immutable six-run
force-step package produced by `molsimflow.workflows.tpcl_force_step`.

## Required matrix

The package must contain `f0_shared`, `f8e-5_x`, and `f8e-5_y` production runs
for both `ch3_only` and `mixed291`. Every run must have a passing
`RUN-RESULT.txt`, a passing `VALIDATION.json`, and terminal step `36400000`.

## Analysis order

1. Freeze the contact-water population size from the same-surface F0 parent
   frame, then build substrate-fixed 10% tail-mean footprint edges while
   preserving LAMMPS image flags.
2. Subtract the same-surface `f0_shared` trace from each driven trace.
3. Smooth over a 1 ps centered window and freeze persistent start, peak, and
   stall steps from leading/trailing-edge kinematics.
4. Match non-event controls within the same branch and output-cadence phase.
5. Only after event selection, evaluate SiOH--water H-bond anchoring and TPCL
   water-network turnover from the full-reference stream.

This ordering prevents an anchor metric from selecting the events used to test
that same metric. `ch3_only` is the no-SiOH negative control. Time blocks are
descriptive and are not independent replicas.

## Invocation

```bash
PYTHONPATH=/path/to/code_snapshot \
python -m molsimflow.postprocess.tpcl_force_step_analysis \
  --package-root /path/to/four_interface_tpcl_high_frequency_force_on_v1_r3 \
  --output-dir /path/to/fresh/high_frequency_tpcl_mechanism_v1
```

The output directory is immutable: the command fails if it already exists. It
contains the contract, source manifest, per-branch kinematics and anchor
tables, paired global response, frozen events and controls, figures, review,
validation summary, and SHA256 manifest.

## Scientific boundary

A passing output is a single-window paired event-association result. It is not
an independent-replica causal test, a free-energy barrier, a converged event
rate, or a transferable mobility tensor.
