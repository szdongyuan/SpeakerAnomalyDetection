"""Controller-facing analysis input API; importing it starts no runtime work."""

from .contracts import (
    AnalysisBatch,
    AnalysisChannel,
    AnalysisInputError,
    AnalysisInputIssue,
    AnalysisInstance,
    AnalysisTask,
)
from .input_adapter import build_analysis_batch

__all__ = [
    "AnalysisBatch", "AnalysisChannel", "AnalysisInputError", "AnalysisInputIssue",
    "AnalysisInstance", "AnalysisTask", "build_analysis_batch",
]
