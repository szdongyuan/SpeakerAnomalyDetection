import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base.pre_processing.alignment_processing import AlignmentProcessing


def test_align_play_and_rec_preserves_recorded_float64_when_padding(monkeypatch):
    monkeypatch.setattr(
        AlignmentProcessing,
        "gcc_phat",
        staticmethod(lambda stimulus, recorded: (2, None, None)),
    )

    stimulus = np.arange(5, dtype=np.float32)
    recorded = np.array([10.0, 11.0, 12.0], dtype=np.float64)

    aligned = AlignmentProcessing.align_play_and_rec_data_using_gccphat(stimulus, recorded)

    assert aligned.dtype == np.float64
    np.testing.assert_array_equal(aligned, np.array([12.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))


def test_align_play_and_rec_preserves_recorded_float32(monkeypatch):
    monkeypatch.setattr(
        AlignmentProcessing,
        "gcc_phat",
        staticmethod(lambda stimulus, recorded: (0, None, None)),
    )

    stimulus = np.arange(3, dtype=np.float32)
    recorded = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    aligned = AlignmentProcessing.align_play_and_rec_data_using_gccphat(stimulus, recorded)

    assert aligned.dtype == np.float32
    np.testing.assert_array_equal(aligned, recorded)
