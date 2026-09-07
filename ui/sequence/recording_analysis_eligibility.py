"""Frozen, request-scoped policy for fast recording overlap admission."""
from collections.abc import Mapping


FAST_OVERLAP_ANALYSES = frozenset(("SPL", "FBA", "SPEC"))


def enabled_analysis_identifiers(config_snapshot):
    """Freeze enabled analysis types from one request/context configuration."""
    snapshot = config_snapshot if isinstance(config_snapshot, Mapping) else {}
    analysis = snapshot.get("analysis_config", snapshot)
    if not isinstance(analysis, Mapping):
        return ("<INVALID_ANALYSIS_CONFIG>",)
    display = analysis.get("display_sequence", ())
    if not isinstance(display, (list, tuple)):
        return ("<INVALID_DISPLAY_SEQUENCE>",)
    identifiers = []
    for key in display:
        item = analysis.get(key)
        declared_type = item.get("type") if isinstance(item, Mapping) else None
        identifier = declared_type if isinstance(item, Mapping) else key
        normalized = str(identifier or key or "<UNKNOWN_ANALYSIS>").strip().upper()
        if (isinstance(declared_type, str)
                and declared_type.strip().upper() == "EXCEL"):
            continue
        identifiers.append(normalized or "<UNKNOWN_ANALYSIS>")
    return tuple(identifiers)


def fast_recording_overlap_eligible(new_identifiers, unfinished_identifiers):
    """Allow fast overlap iff every new and unfinished identifier is approved."""
    identifier_sets = (tuple(new_identifiers), *(
        tuple(identifiers) for identifiers in unfinished_identifiers))
    return all(identifier in FAST_OVERLAP_ANALYSES
               for identifiers in identifier_sets for identifier in identifiers)
