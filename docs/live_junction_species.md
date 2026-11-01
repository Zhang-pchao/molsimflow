# Latest-frame species checks for periodic chemical junctions

The reusable helpers in molsimflow.io.live_snapshot capture a fixed byte prefix
of a growing trajectory and select its last complete orthorhombic dump frame.
Only an incomplete EOF tail is tolerated. Interior format corruption and other
zstd errors fail closed. The compressed prefix can be deleted after its size,
SHA256 and selected-frame identity have been recorded.

The helpers in molsimflow.postprocess.junction_species provide periodic stripe
inspection regions, median framework-translation correction, explicit bond-ID
comparison and regional count conservation. They contain no case paths or
scheduler assumptions and do not import legacy scripts.

Use initial material identities for surface sites and complete pendant groups.
Classify molecules using the full periodic geometry before assigning spatial
regions; cropping the atoms into separate half-boxes can fabricate broken
molecules or detached terminal groups. In contrast, fluid spatial counts should
follow the current molecular anchor after correcting framework translation.

An AB stripe has an internal seam and an outer periodic seam. A configurable
inspection half-width is a diagnostic convention, not a measured physical
interface thickness. Bond-identity comparison supplements coordination counts:
a bond exchange can preserve each Si coordination number while changing the
framework connectivity.

Project wrappers may reuse a separately audited surface-species analyzer and
record its exact source hash. Keep that integration and all run paths outside
the reusable package. Execution success, classifier warnings and physical
acceptance should be separate states.

For a repeatable latest-frame workspace, publish a complete staged bundle by
atomically switching a stable current symlink between two fixed output slots.
Delete the obsolete successful bundle only after publication. Keep a compact
update journal, source-prefix and frame hashes, and explicit failure evidence;
do not silently replace a failed refresh with an older result labelled current.

The focused standard-library regression suite is:

    PYTHONPATH=src python tests/test_live_junction_species.py
