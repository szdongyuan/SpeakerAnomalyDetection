"""Saved analysis artifact naming shared by reports and the recording viewer."""

import re
from pathlib import Path

CHANNEL_IMAGE_RE = re.compile(
    r"^(?P<key>.+)_CH(?P<channel>[1-9][0-9]*)(?:\((?P<label>[^)]*)\))?$"
)
CHANNEL_ARTIFACT_RE = re.compile(
    r"^CH(?P<channel>[1-9][0-9]*)(?:\((?P<label>[^)]*)\))?(?=$|_)"
)
CHANNEL_NAME_RE = re.compile(r"^CH(?P<channel>[1-9][0-9]*)$")
NATURAL_PART_RE = re.compile(r"([0-9]+)")
JUDGEMENT_STATUS_FILE = "analysis_judgement.csv"
CSV_ROLES = (
    ("总体声压级", "SPL", True),
    ("实时声压级", "SPL", False),
    ("频段能量", "FBA", False),
    ("FFT频谱", "FFT", False),
    ("响度汇总", "LOUD", True),
    ("响度时间曲线", "LOUD", False),
    ("尖锐度汇总", "SHRP", True),
    ("尖锐度时间曲线", "SHRP", False),
    ("粗糙度汇总", "ROUGH", True),
    ("粗糙度时间曲线", "ROUGH", False),
)

def parse_csv_artifact(path: Path):
    stem = path.stem
    for role, analysis_type, _is_scalar in CSV_ROLES:
        if analysis_type in {"SHRP", "ROUGH"}:
            match = re.fullmatch(rf"(.+)_{role}_CH[1-9][0-9]*", stem)
            if match:
                return match.group(1), analysis_type, role
        suffix = f"_{role}"
        if stem.endswith(suffix) and len(stem) > len(suffix):
            key = stem[: -len(suffix)]
            return key, analysis_type, role
    return None


def infer_analysis_type(key: str) -> str:
    normalized = str(key or "").upper()
    hints = (
        ("SHRP", "SHRP"), ("尖锐度", "SHRP"),
        ("ROUGH", "ROUGH"), ("粗糙度", "ROUGH"),
        ("LOUD", "LOUD"), ("响度", "LOUD"),
        ("SPEC", "Spec"),
        ("频谱分析", "Spec"),
        ("SPL", "SPL"),
        ("声压级", "SPL"),
        ("FBA", "FBA"),
        ("频段能量", "FBA"),
        ("FFT", "FFT"),
        ("快速傅里叶", "FFT"),
    )
    for hint, analysis_type in hints:
        if hint in normalized:
            return analysis_type
    return ""


def split_segment_channel(value):
    match = re.fullmatch(
        r"(?P<label>(?:输出负载|时间).*)_(?P<channel>CH[1-9][0-9]*(?:\(.*\))?)",
        str(value or "").strip(),
    )
    return (match.group("label"), match.group("channel")) if match else ("", "")
