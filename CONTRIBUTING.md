# Contributing

The repository keeps two long-lived branches:

- `devel` is the integration branch for code, tests, and documentation;
- `main` contains changes that have passed the repository checks.

Work on `devel`, keep commits focused, and merge into `main` after running:

```bash
python scripts/audit_public_repo.py
python -m compileall -q src scripts tests
PYTHONPATH=src python -m pytest -q
```

Source code, comments, command help, tests, documentation, and commit messages
should be in English. Do not commit private paths, cluster credentials, real
trajectories, generated results, or files from `legacy_sources/`.

New non-trivial numerical or parsing behavior should include one small
synthetic test. Prefer descriptive module and function names, explicit user
inputs, standard-library features, and existing package helpers over new
frameworks or parallel implementations.
