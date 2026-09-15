"""Shared overall-SPL CSV column names and historical unit compatibility."""

_PREFIXES = ("总体声压级", "总体下限", "总体上限")
_WEIGHTINGS = "ABCDZ"


def overall_spl_csv_columns(weighting):
    normalized = str(weighting or "Z").strip().upper()
    if normalized == "NONE":
        normalized = "Z"
    if normalized not in _WEIGHTINGS or len(normalized) != 1:
        raise ValueError(f"不支持的 SPL 计权方式：{weighting}")
    return tuple(f"{prefix}dB({normalized})" for prefix in _PREFIXES)


def resolve_overall_spl_csv_columns(fieldnames):
    headers = set(fieldnames or ())
    choices = [(tuple(f"{p}dB" for p in _PREFIXES), "dB(A)")]
    choices.extend((overall_spl_csv_columns(w), f"dB({w})") for w in _WEIGHTINGS)
    matches = [(columns, unit) for columns, unit in choices if columns[0] in headers]
    if len(matches) != 1:
        raise ValueError("总体声压级数值列缺失或存在多个计权列")
    columns, unit = matches[0]
    for other, _ in choices:
        if other != columns and any(column in headers for column in other[1:]):
            raise ValueError("总体声压级与上下限列的计权不一致")
    # Historical unqualified dB headers are treated as A by the agreed compatibility rule.
    return columns, unit
