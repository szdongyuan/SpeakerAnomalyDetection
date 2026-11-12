import hashlib

import numpy as np
import pytest

from base.core_algorithm.sound_quality.roughness import (
    compute_roughness_daniel_weber,
    compute_roughness_mosqito_dw,
)


SAMPLE_RATE = 48000
DURATION_S = 0.6
OVERLAP = 0.5

EXPECTED_ROUGHNESS_CASES = {
    "pure_1khz": {
        "frames": 5,
        "roughness_asper": [
            0.001298066552354,
            0.001298066552384,
            0.001298066552367,
            0.001298066552397,
            0.001298066552463,
        ],
        "specific_digest_15dp": "70cf088750668d2e59d18f1ef1a1ecaa95fd755f0d73b33b9607b7f519f43e23",
        "specific_sum_by_frame": [
            0.005192266209417,
            0.005192266209535,
            0.005192266209467,
            0.005192266209589,
            0.005192266209851,
        ],
        "summary": {
            "r_mean_asper": 0.001298066552392956,
            "r_max_asper": 0.0012980665524627672,
        },
    },
    "am_70hz_1khz": {
        "frames": 5,
        "roughness_asper": [
            0.553998364427965,
            0.553998364427946,
            0.553998364427989,
            0.553998364427948,
            0.553998364427941,
        ],
        "specific_digest_15dp": "a06bc2de316893be812fb7c635bf6b43ad56609025f7266ca0ca6d627b18d55b",
        "specific_sum_by_frame": [
            2.21599345771186,
            2.215993457711786,
            2.215993457711955,
            2.215993457711794,
            2.215993457711764,
        ],
        "summary": {
            "r_mean_asper": 0.553998364427958,
            "r_max_asper": 0.5539983644279887,
        },
    },
    "two_tone": {
        "frames": 5,
        "roughness_asper": [
            0.001936176241971,
            0.001936176240344,
            0.001936176240379,
            0.001936176240367,
            0.001936176243951,
        ],
        "specific_digest_15dp": "d3c7dafadce1836c28a12320d8a7fa388f9d7a86ba930bcc0983ce9e76e16dda",
        "specific_sum_by_frame": [
            0.007744704967885,
            0.007744704961377,
            0.007744704961518,
            0.007744704961467,
            0.007744704975805,
        ],
        "summary": {
            "r_mean_asper": 0.0019361762414025912,
            "r_max_asper": 0.001936176243951268,
        },
    },
}


def _time_axis() -> np.ndarray:
    return np.arange(int(SAMPLE_RATE * DURATION_S), dtype=np.float64) / SAMPLE_RATE


def _signal(case_name: str) -> np.ndarray:
    t = _time_axis()
    if case_name == "pure_1khz":
        return np.sqrt(2.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    if case_name == "am_70hz_1khz":
        return (1.0 + 0.5 * np.sin(2.0 * np.pi * 70.0 * t)) * np.sin(
            2.0 * np.pi * 1000.0 * t
        )
    if case_name == "two_tone":
        return 0.8 * np.sin(2.0 * np.pi * 700.0 * t) + 0.4 * np.sin(
            2.0 * np.pi * 1200.0 * t
        )
    raise KeyError(case_name)


def _specific_digest(values: np.ndarray) -> str:
    rounded = np.round(np.ascontiguousarray(values, dtype=np.float64), 15)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


@pytest.mark.parametrize("case_name", sorted(EXPECTED_ROUGHNESS_CASES))
def test_roughness_daniel_weber_regression(case_name):
    expected = EXPECTED_ROUGHNESS_CASES[case_name]

    result = compute_roughness_daniel_weber(
        _signal(case_name),
        SAMPLE_RATE,
        overlap=OVERLAP,
    )

    assert result.time_s.size == expected["frames"]
    np.testing.assert_allclose(
        result.roughness_asper,
        np.asarray(expected["roughness_asper"], dtype=np.float64),
        rtol=0.0,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        np.sum(result.specific_roughness_asper_per_bark, axis=0),
        np.asarray(expected["specific_sum_by_frame"], dtype=np.float64),
        rtol=0.0,
        atol=5e-15,
    )
    assert _specific_digest(result.specific_roughness_asper_per_bark) == expected["specific_digest_15dp"]

    assert result.summary.r_mean_asper == pytest.approx(expected["summary"]["r_mean_asper"], abs=5e-15)
    assert result.summary.r_max_asper == pytest.approx(expected["summary"]["r_max_asper"], abs=5e-15)


def test_roughness_project_backend_matches_mosqito_reference_when_available():
    pytest.importorskip("mosqito")
    signal = _signal("am_70hz_1khz")

    project = compute_roughness_daniel_weber(signal, SAMPLE_RATE, overlap=OVERLAP)
    reference = compute_roughness_mosqito_dw(signal, SAMPLE_RATE, overlap=OVERLAP)

    np.testing.assert_allclose(project.roughness_asper, reference.roughness_asper, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(
        project.specific_roughness_asper_per_bark,
        reference.specific_roughness_asper_per_bark,
        rtol=0.0,
        atol=1e-11,
    )
