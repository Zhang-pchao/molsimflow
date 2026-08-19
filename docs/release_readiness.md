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
python -m compileall -q src scripts tests
PYTHONPATH=src python -m pytest -q
```

If `pytest` is unavailable in the active cluster environment, run the smoke
commands documented in the migration notes for the changed workflow and record
that `pytest` was not available.

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
