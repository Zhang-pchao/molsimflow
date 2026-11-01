# Release Readiness

## Current State

`molsimflow` is maintained as an engineering repository for continued public
GitHub development. The tracked tree contains package code, tests,
templates, and migration documentation.  The raw legacy snapshot stays under
`legacy_sources/`, which is ignored by Git.

## Required Checks

Run from the repository root:

```bash
python scripts/audit_public_repo.py
python scripts/generate_module_catalog.py --check
python -m compileall -q src scripts tests
PYTHONPATH=src python -m pytest -q
```

Release promotion requires the full suite on Python 3.9 and 3.12. A missing
pytest installation or failed check is not replaced by a smoke test. Development
smokes may diagnose an environment but do not satisfy the release gate.

## Public-Repo Gate

Before each public update:

- run `python scripts/audit_public_repo.py`;
- verify `git status --ignored --short` shows `legacy_sources/` ignored;
- inspect any new docs/examples for private absolute paths;
- keep generated outputs, trajectories, figures, CSV tables, logs, and backup
  files out of Git;
- decide whether example configs should remain placeholders or move into a
  separate private run repository.

## Repository Scope

Public commits may include:

- `pyproject.toml`, `README.md`, `.gitignore`, `AGENTS.md`;
- `src/molsimflow` package code;
- `scripts/audit_public_repo.py` and `scripts/run_molsimflow.py`;
- `templates/` with only public placeholders;
- `docs/` migration and usage notes;
- `tests/` synthetic unit tests.

Do not include `legacy_sources/` or real simulation outputs.

The repository uses the MIT license. Long-lived development is limited to
`devel` for integration and `main` for validated changes.


## Artifact and adoption gate

Build from the exact qualified commit in external build storage. Record package
version, source commit, artifact SHA256, Python version and dependencies. Test
the installed artifact from outside source directories and verify import origin.
Deploy the same artifact to fresh versioned prefixes; preserve the previous
prefix and consumer entrypoint for rollback. Existing scientific runs retain
an explicit runtime until their owning project accepts a change.

The module catalog describes implementation scope and APIs; passing tests does
not establish convergence, material transferability or causal interpretation.
New material adapters and changes to scientific definitions require their own
qualification. Branch/worktree deletion additionally requires recoverability
and a consumer/active-writer check, not merely a successful merge.
