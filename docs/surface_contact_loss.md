# Physical interpretation of contact support loss

The existing supported_material_regions observer now also exposes all-gas near
membership and nested all-gas/main-gas shell counts. Its original support,
accepted counts and molecule membership are unchanged; diagnostic fields have
optional defaults for backward-compatible construction. The same periodic
geometry is used; no duplicate region or gas-connectivity implementation exists.

diagnose_region_support_losses reuses prospective_count_losses with a one-frame
search and no dwell requirement. It enumerates all adjacent supported-to-unsupported
region transitions without crossing source or cadence gaps. Count hierarchy is
validated; missing geometry is not zero. The exclusive first-failed-frame classes
separate geometric emptiness, near gas outside the contact shell, non-main shell
contact, subthreshold main count, and local connectivity failure.

Near gas does not imply actual surface contact. Zero geometric count at one saved
frame does not prove physical evacuation or uninterrupted absence. Overlapping
regions and repeated transitions are correlated and do not define a kinetic rate.
