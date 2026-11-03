"""Calibration calculation and post-release adapter on the capture owner."""
import sys
import time

import numpy as np

from base.vkinging_capture import capture_vkinging, _checkpoint
from base.vkinging_service import CaptureResult, _exception_text
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis


def calculate_calibration_factor(raw, sample_rate, target_spl):
    """Apply the existing RMS1201/middle-interval algorithm to raw volts."""
    if type(target_spl) is not int or target_spl not in (94, 114):
        raise ValueError("calibration target must be 94 or 114 dB")
    raw = np.asarray(raw, dtype=float)
    if (raw.ndim != 1 or len(raw) != sample_rate * 10 or len(raw) < 1201
            or not np.isfinite(raw).all()):
        raise ValueError("calibration requires complete finite ten second mono voltage")
    step = len(raw) // 3
    center = len(raw) // 2
    middle = raw[center - step:center + step]
    if np.sqrt(np.mean((middle - np.mean(middle)) ** 2)) <= 1e-10:
        raise ValueError("calibration requires AC RMS above 1e-10 V")
    smooth = AudioThdFrequencyResponseAnalysis().spl_calculation(raw, method="rms", window_size=1201)
    center = len(smooth) // 2
    average = float(np.mean(smooth[center - step:center + step]))
    factor = float(10 ** ((target_spl - average) / 20))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("calibration factor must be finite and positive")
    return factor


def capture_calibration(sdk, request, control, *, clock=time.monotonic):
    """Compose capture's file finalizer; never create a second native owner."""
    if request.calibration_target_spl is None:
        raise ValueError("calibration target SPL is required")
    captured = capture_vkinging(sdk, request, control, clock=clock)

    def finalize(control):
        prepared = captured.finalize(control)
        import soundfile as sf
        closed = True
        cleanup_reported = False
        try:
            _checkpoint(control)
            closed = False
            try:
                stream = sf.SoundFile(prepared.value["path"], mode="r")
            except sf.LibsndfileError:
                # With this path/r-only call, libsndfile's open error is raised
                # when sf_open returns NULL: no file handle was acquired.
                closed = True
                raise
            try:
                if (stream.samplerate != request.sample_rate or stream.channels != 1
                        or len(stream) != request.target_samples):
                    raise ValueError("calibration file differs from frozen request")
                blocks = []
                for block in stream.blocks(blocksize=65536, dtype="float64", always_2d=False):
                    _checkpoint(control)
                    blocks.append(block)
                raw = np.concatenate(blocks) if blocks else np.array([])
            finally:
                primary = sys.exception()
                try:
                    stream.close()
                except Exception as close_error:
                    # soundfile.close is the external file boundary. Retain
                    # the read error and mark the newly opened handle uncertain.
                    diagnostic = ("calibration file close failed: " + _exception_text(close_error)
                                  + "; owned paths: " + ", ".join(prepared.value.get("owned_paths", (prepared.value["path"],))))
                    control.report_cleanup_failure(diagnostic)
                    cleanup_reported = True
                    if primary is None:
                        raise
                    primary.add_note(diagnostic)
                else:
                    closed = True
            _checkpoint(control)
            factor = calculate_calibration_factor(raw, request.sample_rate, request.calibration_target_spl)
            _checkpoint(control)
            value = {**dict(prepared.value), "factor": factor, "standard_spl": request.calibration_target_spl}
            # Internal calibration keeps the released owned partial and never
            # publishes an ordinary recording file. Service validates success.
            return CaptureResult(value, resources_released=True)
        finally:
            if sys.exception() is not None:
                if closed:
                    control.confirm_released()
                elif not cleanup_reported:
                    # An unexpected constructor exception has no returned
                    # handle whose closure can be proved. Do not claim release.
                    control.report_cleanup_failure(
                        "calibration file open/release uncertain: " + _exception_text(sys.exception())
                        + "; owned paths: " + ", ".join(prepared.value.get("owned_paths", (prepared.value["path"],))))

    return CaptureResult(captured.value, resources_released=captured.resources_released,
                         cleanup_error=captured.cleanup_error, finalize=finalize)
