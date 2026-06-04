import re

from consts.frequency_stepped_consts import FREQUENCY_STEPPED_METHOD


_FREQUENCY_STEPPED_ALIASES = {
    FREQUENCY_STEPPED_METHOD,
    "frequency_step",
    "frequency_steps",
    "frequencystepped",
    "frequencystep",
    "frequencysteps",
}


def _alias_key(method):
    return re.sub(r"[\s_\-()]+", "", str(method).strip().lower())


def normalize_stimulus_method(method):
    stripped = str(method).strip()
    normalized = stripped.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"chirp", "step", "noise"}:
        return normalized
    if normalized in {FREQUENCY_STEPPED_METHOD, "frequency_step", "frequency_steps"}:
        return FREQUENCY_STEPPED_METHOD
    if _alias_key(stripped) in _FREQUENCY_STEPPED_ALIASES:
        return FREQUENCY_STEPPED_METHOD
    return stripped


def analysis_stimulus_method(method):
    normalized = normalize_stimulus_method(method)
    if normalized == "chirp":
        return "chirps"
    if normalized == "step":
        return "steps"
    return normalized
