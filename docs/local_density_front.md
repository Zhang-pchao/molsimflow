# Local water-density front observer

The pure helpers in local_density_front build an XYZ density field in a fixed material frame, then measure independent azimuthal interface radii over a supplied height band. A star-shaped low-water-density cavity is required. The observer does not fit a circle or impose a common radius.

The gas-to-water interface normal includes both height and azimuthal slopes. The reported angle is an apparent gas-side tangent angle at the fit band's center; the boundary is a linear extrapolation to the caller's reference plane. Its normal uses the azimuthal slope of that extrapolated boundary. Neither output establishes molecular gas-solid contact or a Young equilibrium angle.

The caller must supply current water identities, a registered solid reference, independent real-contact support and topology checks. Missing radial crossings and center-wet sectors remain invalid. A closed complete boundary is required for normal_displacements; missing sectors are never joined automatically. The latter intersects each old in-plane outward normal with the new polygon, giving signed normal displacement in the common material frame.

Tests include a spherical cavity with an analytic angle, an anisotropic ellipsoid with azimuthally varying normal, periodic translation of the density grid, absence of a cavity in uniform water, closed-boundary translation/expansion and rejection of incomplete contours. These checks validate numerical geometry only. Physical resolution, height-band sensitivity, water-film separation and observational uncertainty require dataset-specific qualification.

supported_slab_contact now accepts periodic_z=False explicitly. It retains physical open-Z distances and face assignment and pads Z only for the existing molecule/component helpers. The default PPP path is unchanged. Regression tests cover PPP/PPF equivalence away from the seam, no false contact/connection through open Z and translation invariance. Contact and connectivity cutoffs are not changed by the boundary option.
