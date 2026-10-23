# Contributing

The repository keeps two long-lived branches:

- `devel` is the integration branch for code, tests, and documentation;
- `main` contains changes that have passed the repository checks.

Use short-lived feature branches based on the current integration history.
`pimd-postprocess` is a controlled PIMD refactoring branch, not a permanent
alternative version of the package. Keep its changes limited to PIMD and
necessary shared primitives, merge tested batches into `devel`, and merge
`devel` back when integration advances. Preserve shared history; do not rebase
or duplicate already-integrated commits. Retire the active PIMD branch after
its integration unless a continuing maintenance need has an explicit owner.

Package boundaries isolate domains. Branches track review and release state.
Preserve dirty worktrees and unrelated files; never use cleanup or forced ref
updates as an integration strategy.

Keep commits focused, and merge a qualified `devel` release into `main` after
running the checks on Python 3.9 and 3.12:

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
