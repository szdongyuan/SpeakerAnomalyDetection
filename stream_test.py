"""
Play a WAV and record with one sounddevice duplex Stream on Windows WASAPI.

Uses the **WASAPI host API's** default input and output device indices (not
``sd.default.device``, which may point at MME/DirectSound). No fallbacks: wrong
host API, digital device name, channel count, or sample-rate mismatch raises
immediately. Mono WAV files open the **output** side as 2 channels when the
device reports stereo, because many WASAPI duplex setups reject a single
output channel (``PaErrorCode -9998``); the mono sample is duplicated to L/R.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

# WAV relative to this script directory (override with argv[1])
STIMULUS_RELATIVE = (
    Path("audio_data")
    / "stimulus"
    / ("stimulus_chirps_1_True_chirp_mirror_log_80_2000_4.0_1_1_RMS_2.0_0.3481_48000.wav")
)

_DIGITAL_NAME_MARKERS = (
    "digital audio",
    "s/pdif",
    "spdif",
    "hdmi",
    "display audio",
    "数字音频",
)


class StreamTestError(RuntimeError):
    pass


@dataclass(frozen=True)
class DuplexPlan:
    """WASAPI default devices plus PortAudio channel layout for ``sd.Stream``."""

    input_id: int
    output_id: int
    input_name: str
    output_name: str
    channels_in: int
    channels_out: int
    file_channels: int


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_wav_path() -> Path:
    return _script_dir() / STIMULUS_RELATIVE


def _is_digital(name: str) -> bool:
    n = name.lower()
    return any(m in n for m in _DIGITAL_NAME_MARKERS)


def _wasapi_hostapi() -> tuple[int, dict]:
    for i, h in enumerate(sd.query_hostapis()):
        if h.get("name") == "Windows WASAPI":
            return i, h
    raise StreamTestError("Windows WASAPI is not available in PortAudio.")


def _wasapi_default_duplex_indices() -> tuple[int, int]:
    """
    Global device indices for WASAPI default input / output.

    sd.default.device follows Windows *default* host API (often MME for mic);
    duplex must open WASAPI endpoints, so we use the WASAPI host API defaults.
    """
    _wasapi_index, hostapi = _wasapi_hostapi()
    try:
        in_id = hostapi["default_input_device"]
        out_id = hostapi["default_output_device"]
    except (KeyError, TypeError) as e:
        raise StreamTestError(f"WASAPI host API info missing defaults: {e}") from e
    if in_id is None or out_id is None:
        raise StreamTestError(f"WASAPI default input/output not set (input={in_id!r}, output={out_id!r}).")
    return int(in_id), int(out_id)


def _duplex_stream_channel_counts(file_ch: int, din: dict, dout: dict) -> tuple[int, int]:
    """
    The Dongyuan WASAPI input endpoint exposes 6 channels and rejects smaller
    input counts, so open input with the device channel count and save channel 1.
    Mono WAV output still opens as stereo when needed and duplicates to L/R.
    """
    mic_max = int(din["max_input_channels"])
    out_max = int(dout["max_output_channels"])
    if file_ch > mic_max:
        raise StreamTestError(
            f"WASAPI default input {din['name']!r}: file has {file_ch} ch, max_input_channels={mic_max}."
        )
    if file_ch > out_max:
        raise StreamTestError(
            f"WASAPI default output {dout['name']!r}: file has {file_ch} ch, max_output_channels={out_max}."
        )
    ch_in = mic_max
    ch_out = file_ch
    if ch_out == 1 and out_max >= 2:
        ch_out = 2
    return ch_in, ch_out


def _require_wasapi_non_digital_duplex(file_ch: int, file_sr: float) -> DuplexPlan:
    wasapi, _hostapi = _wasapi_hostapi()
    devices = sd.query_devices()
    in_id, out_id = _wasapi_default_duplex_indices()

    din = devices[in_id]
    dout = devices[out_id]

    if din["hostapi"] != wasapi:
        raise StreamTestError(f"WASAPI default input is not on WASAPI host: {din['name']!r}")
    if dout["hostapi"] != wasapi:
        raise StreamTestError(f"WASAPI default output is not on WASAPI host: {dout['name']!r}")

    if _is_digital(str(din["name"])):
        raise StreamTestError(f"WASAPI default input is rejected as digital: {din['name']!r}")
    if _is_digital(str(dout["name"])):
        raise StreamTestError(f"WASAPI default output is rejected as digital: {dout['name']!r}")

    ch_in, ch_out = _duplex_stream_channel_counts(file_ch, din, dout)

    def _sr_ok(dev: dict) -> bool:
        try:
            dsr = float(dev["default_samplerate"])
        except (TypeError, ValueError) as e:
            raise StreamTestError(f"Invalid default_samplerate on {dev['name']!r}: {e}") from e
        if abs(dsr - file_sr) > 1e-3:
            raise StreamTestError(f"Device {dev['name']!r} default_samplerate={dsr} must equal file {file_sr} Hz.")
        return True

    _sr_ok(din)
    _sr_ok(dout)

    return DuplexPlan(
        input_id=in_id,
        output_id=out_id,
        input_name=str(din["name"]),
        output_name=str(dout["name"]),
        channels_in=ch_in,
        channels_out=ch_out,
        file_channels=file_ch,
    )


def run_play_record(wav_path: Path, out_wav: Path | None = None) -> Path:
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise StreamTestError(f"WAV not found: {wav_path}")

    out_wav = Path(out_wav) if out_wav is not None else wav_path.parent / f"{wav_path.stem}_recorded.wav"

    with sf.SoundFile(str(wav_path)) as sf_in:
        ch = int(sf_in.channels)
        sr = float(sf_in.samplerate)
        if ch < 1 or sr <= 0:
            raise StreamTestError("Invalid WAV channels or sample rate.")

        plan = _require_wasapi_non_digital_duplex(ch, sr)

        chunks: list[np.ndarray] = []
        done = threading.Event()
        errors: list[BaseException] = []

        def callback(indata, outdata, frames, time, status):
            try:
                if status:
                    print(f"status: {status}")
                chunks.append(np.copy(indata))

                block = sf_in.read(frames, dtype="float32", always_2d=True)
                block = np.asarray(block, dtype=np.float32)
                if block.ndim == 1:
                    block = block.reshape(-1, 1)

                n = block.shape[0]
                fc = plan.file_channels
                so = plan.channels_out
                if n:
                    if fc == 1 and so == 2:
                        m = block[:, 0]
                        outdata[:n, 0] = m
                        outdata[:n, 1] = m
                    else:
                        c = min(block.shape[1], outdata.shape[1])
                        outdata[:n, :c] = block[:, :c]
                        if c < outdata.shape[1]:
                            outdata[:n, c:].fill(0.0)
                if n < frames:
                    outdata[n:, :].fill(0.0)
                    done.set()
                    raise sd.CallbackStop
            except sd.CallbackStop:
                raise
            except BaseException as e:
                errors.append(e)
                outdata[:, :].fill(0.0)
                done.set()
                raise sd.CallbackStop from e

        stream = sd.Stream(
            device=(plan.input_id, plan.output_id),
            samplerate=sr,
            channels=(plan.channels_in, plan.channels_out),
            dtype=("float32", "float32"),
            callback=callback,
        )

        print(f"file: {wav_path}  ({sr} Hz, {ch} ch file)")
        print(
            f"WASAPI duplex open: in={plan.channels_in} ch, out={plan.channels_out} ch "
            f"(device: in {plan.input_name!r} [{plan.input_id}], out {plan.output_name!r} [{plan.output_id}])"
        )
        print(f"record -> {out_wav}")

        try:
            with stream:
                done.wait()
        finally:
            pass

        if errors:
            raise StreamTestError(f"callback: {errors[0]}") from errors[0]

        if not chunks:
            raise StreamTestError("No input samples captured.")

        recorded = np.vstack(chunks)
        first_channel = recorded[:, :1]
        sf.write(str(out_wav), first_channel, int(sr), subtype="FLOAT")

    print(f"wrote {first_channel.shape[0]} samples from input channel 1 -> {out_wav}")
    return out_wav


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) > 2:
        print("usage: stream_test.py [path/to.wav] [path/to_recorded.wav]", file=sys.stderr)
        return 2
    wav = Path(argv[0]) if argv else default_wav_path()
    out_rec = Path(argv[1]) if len(argv) > 1 else None
    try:
        run_play_record(wav, out_rec)
    except StreamTestError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
