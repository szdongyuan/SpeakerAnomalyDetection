"""Specific-loudness assessment reference curves; not algorithm coefficients."""

# SSTS Specific Loudness Reference Lines (from Seat SSTS Revision 40, Table 46 / Figure 6)
# Format: list of (Bark, sone/Bark) breakpoints — linearly interpolated between points.
# Units: Loudness in sone GF (free-field).
SSTS_SPECIFIC_LOUDNESS_REF_LINES = {
    "ref1": [(0, 0.33), (4, 0.33), (6, 0.18), (7, 0.18), (16, 0.18), (18, 0.1375), (24, 0.01)],
    "ref2": [(0, 0.50), (4, 0.50), (6, 0.22), (7, 0.22), (16, 0.22), (18, 0.1675), (24, 0.01)],
    "ref3": [(0, 0.40), (4, 0.40), (6, 0.267), (7, 0.20), (16, 0.20), (18, 0.20), (24, 0.01)],
    "ref4": [(0, 0.60), (4, 0.60), (6, 0.18), (7, 0.18), (16, 0.18), (18, 0.1375), (24, 0.01)],
}
