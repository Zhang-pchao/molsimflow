# Publicization Checklist

Run this checklist before publishing or opening a GitHub repository.

- Search for private absolute paths, cluster filesystem roots, and project-specific case directories.
- Search for scheduler commands embedded in Python logic: `sbatch`, `qsub`, `module load`, and `conda activate`.
- Keep private run directories and generated data out of Git.
- Remove backup files and copied historical variants from migration candidates.
- Replace personal defaults with CLI arguments, environment variables, or documented example config files.
- Add tests or small fixtures for each migrated workflow.
- Keep examples small enough for a normal checkout.
- Decide on a license before public release.

Repository audit:

```bash
python scripts/audit_public_repo.py
```

Release-readiness notes are tracked in `docs/release_readiness.md`.
