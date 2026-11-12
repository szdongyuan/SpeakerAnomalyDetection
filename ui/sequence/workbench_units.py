"""Shared display units for workbench plots and station results."""


def format_workbench_spl_unit(weighting):
    weighting = str(weighting or "Z").upper()
    if weighting in ("NONE", "Z（NONE）"):
        weighting = "Z"
    return f"dB({weighting})"
