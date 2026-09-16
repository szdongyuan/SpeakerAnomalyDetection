"""Small, deterministic benchmark safety and executable smoke contracts."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "tools/benchmark_recording_finalization.py"


def benchmark():
    spec = importlib.util.spec_from_file_location("finalization_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arguments", [
    ["--duration", "nan"], ["--duration", "0"], ["--trim", "-1"],
    ["--rate", "0"], ["--channels", "9"], ["--block-frames", "0"],
    ["--block-frames", "1048577"],
])
def test_reject_invalid_arguments(arguments, tmp_path):
    with pytest.raises(SystemExit):
        benchmark().parse_args([*arguments, "--variant", "optimized",
                                "--output", str(tmp_path / "report.json")])


def test_samples_are_independent_of_block_boundaries():
    module = benchmark()
    expected = module.generated_audio(0, 31, 5)
    actual = np.concatenate([module.generated_audio(0, 7, 5), module.generated_audio(7, 24, 5)])
    np.testing.assert_array_equal(actual, expected)
    assert expected.dtype == np.float32
    assert np.isfinite(expected).all()


def test_counter_measures_actual_partial_reads_and_writes():
    module = benchmark()
    meter = module.IOMeter()
    stream = module.CountedFile(io.BytesIO(b"abcdef"), meter)
    assert stream.read(2) == b"ab"
    assert stream.readinto(bytearray(12)) == 4
    assert stream.read(9) == b""
    assert stream.write(b"123") == 3
    assert meter.bytes["capture"] == {"read_bytes": 6, "write_bytes": 3}


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep me", encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPT), "--output", str(output),
                             "--temp-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep me"
    assert list(tmp_path.iterdir()) == [output]


def test_small_variant_report_and_owned_cleanup(tmp_path):
    output = tmp_path / "report.json"
    sentinel = tmp_path / "user-file.wav"
    sentinel.write_bytes(b"not generated")
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--variant", "optimized", "--source-root",
        str(SCRIPT.parents[1]), "--duration", "0.01", "--trim", "0.001",
        "--output", str(output), "--temp-dir", str(tmp_path),
    ], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["frames"] == 512
    assert report["io"]["production"]["read_bytes"] > 512 * 5 * 4
    assert report["io"]["production"]["write_bytes"] > 512 * 5 * 4
    assert report["io"]["verification"]["read_bytes"] > 0
    assert report["io"]["total"]["read_bytes"] == (
        report["io"]["production"]["read_bytes"] + report["io"]["verification"]["read_bytes"])
    assert report["audio_passes"]["read_prepare"]["read_frames"] == 512
    assert report["sample_sha256"] and report["mono_sha256"]
    assert len(report["waveforms"]) == 5
    assert report["timings_seconds"]["total_completion"] > 0
    assert report["peak_memory_bytes"] is None
    assert sentinel.read_bytes() == b"not generated"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json", "user-file.wav"]
