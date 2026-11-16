# Reusing project postprocessing in the package

Start with [the module catalog](module_catalog.md), the owning architecture
notes and existing tests. Compare units, atom/frame identity, boundary flags,
sample weights, time selection and output schemas before selecting an owner.

A project script can combine several reusable operations. Extract the smallest
coherent operation with explicit inputs and a documented scientific contract.
Keep campaign selection, scheduler control, site paths, input/model identities,
material labels and scientific interpretation in the consuming project.

## Intake record

Keep the following record with the consuming project or private engineering
work. Link a public issue only when its content is suitable for disclosure.

| Item | Evidence to record |
| --- | --- |
| Source | Host, exact script and input identities, checksum, active writer |
| Existing owner | Matching library APIs and the semantic difference to address |
| Public API | Parameters, shapes, units, time/boundary conventions, errors and outputs |
| Compatibility | Existing callers and their defaults, schemas and rendering hooks |
| Qualification | Small synthetic regression and frozen project-input replay |
| Migration | Thin project caller, exact package commit, import origin and rollback |

Prioritize repeated numerical kernels, parsers and identity/alignment checks.
Separate numerical outputs from rendering. Extend an existing owner when its
contract fits. Promote code into a shared layer after independent domains
actually reuse it. Preserve different strictness and scientific conventions
through explicit adapters.

## Example ownership

LAMMPS scalar table parsing lives in `io.lammps_table`. The umbrella adapter in
`postprocess.umbrella_wham.load_window` adds harmonic-bias, time-selection,
clock/energy-conversion and sample-count checks. The existing constant-force
reader delegates to the same raw parser. The numerical WHAM solver accepts
arrays and remains usable with other explicitly prepared fixed-bias inputs.

A consuming umbrella campaign supplies completed windows, actual centers and
spring constants, atom/Hamiltonian identity and restart lineage. Its adapter can
write a portable manifest and invoke the existing module command. The public
manifest needs the physical inputs required for the selected analysis.

## Integration and retirement

Use a short-lived feature branch from current `devel`. Qualify the intended diff
and merge it into `devel`. A qualified release advances `main`. Keep one writer
per shared feature and record an exact-commit handoff when changing hosts.

After project parity passes, reduce the private caller to configuration and
composition. Preserve accepted outputs and the previous runtime entrypoint.
Retire a branch after its history and working changes are accounted for. Remove
a worktree only after its source consumers and untracked/ignored contents have
been checked. A frozen path can stay detached at its recorded commit while its
completed branch name is retired.

Revisit the intake during future feature work and release reviews. Keep changing
candidate lists and project-specific evidence in engineering records so this
page stays short and reusable.
