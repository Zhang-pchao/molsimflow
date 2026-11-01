# Archive lifecycle primitives

`molsimflow.io.archive_lifecycle` provides explicit operations, not an automatic
cleanup daemon. The caller owns host identity, scheduler and process inspection,
input/dependency closure, compression/trajectory qualification, and transport.
Analysis functions must never call cleanup implicitly.

1. `snapshot(root, source_id=..., removable=[...])` hashes the exact regular-file
   inventory after writers stop. Every file is retained unless explicitly named.
   Symlinks, special files and changed-during-hash files are refused.
2. Transfer non-destructively if necessary. Reuse an existing exact archive rather
   than duplicating raw data. Keep partial transfer in an explicit staging root.
3. `verify(manifest, archive_root)` checks exact file set, sizes and SHA256. This
   certifies bytes, not compression validity, full trajectory validity or science.
4. `cleanup_plan(manifest, receipt, allowlist=[...])` freezes the explicit list. It
   cannot name retained inputs, unknown paths, duplicate paths or path escapes.
5. Obtain fresh scheduler/process and dependency evidence. A proof document must
   bind `plan_sha256`, `checked_utc`, `writers_stopped=true`,
   `dependencies_clear=true`, `active_references=[]`, and nonempty evidence records
   containing paths and SHA256. Inspection must include active jobs outside the
   source root, planned consumers, input/restart/model references and symlinks.
   The library does not assert that the caller's dependency audit is complete.
6. Call `cleanup` only for an authorized explicit source root, exact plan SHA256,
   and a new external journal. It requires a proof no older than 15 minutes,
   rehashes all source files, checks source root/file identities, removes only
   listed files with directory descriptors, fsyncs per-file journal records,
   and independently rehashes retained files after exact remaining-set checks.
7. Store the resulting immutable receipts on the archive host; put a small locator
   beside retained source inputs (outside the frozen run inventory). Update the
   project's existing central ledger. Do not claim independent backup redundancy
   when only one archival copy remains.

The source must remain quiescent throughout. Software checks reduce accidental
path/race errors but cannot make concurrent malicious writers safe. A modified
archive after verification also invalidates the caller's release decision.

An interrupted cleanup is not automatically resumed. Keep its journal, inspect
which files remain, revalidate both hosts and dependencies, then explicitly freeze
another same-scope plan if needed. Failed trajectories remain failed after byte-
exact archival; integrity does not imply scientific acceptance.

Targeted verification: `python -m pytest -q tests/test_archive_lifecycle.py`.
