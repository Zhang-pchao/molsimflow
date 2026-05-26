# Repository Instructions

- Write source code, comments, docstrings, CLI help, tests, and commit messages in English.
- Keep reusable library code free of user-specific absolute paths and scheduler environment assumptions.
- Prefer small, typed, testable helpers over monolithic research scripts.
- Keep legacy scripts in `legacy_sources/` only as migration references; do not import from them in package code.
- Add tests for migrated utilities before broad refactors of analysis modules.
- Do not track generated outputs, scheduler logs, backup files, or private case data.
