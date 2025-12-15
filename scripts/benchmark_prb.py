import argparse
import cProfile
import io
import os
import pstats
import time
from typing import List

import numpy as np
from scipy.io import wavfile

# Avoid Matplotlib writing to user home (often blocked / slow in sandboxed environments).
# mosqito may import matplotlib under the hood.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_PROJECT_ROOT, ".mplconfig"))

from base.pre_processing.perceptual_chirp_signal_hd import PerceptualChirpSignalHD


def _parse_harmonic_orders(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec:
        lo_s, hi_s = spec.split("-", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def generate_chirp_with_harmonics(
    sample_rate: int,
    duration_s: float,
    start_hz: float,
    stop_hz: float,
    harmonic_orders: List[int],
    fundamental_amplitude: float = 0.5,
    harmonic_relative_amplitude: float = 0.02,
    noise_std: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = (np.arange(n, dtype=np.float64) / float(sample_rate)).astype(np.float64, copy=False)

    # Linear chirp phase: phi(t) = 2π*(f0*t + 0.5*k*t^2), k=(f1-f0)/T
    k = (stop_hz - start_hz) / float(duration_s)
    phi = 2.0 * np.pi * (start_hz * t + 0.5 * k * t * t)
    y = fundamental_amplitude * np.sin(phi)

    for h in harmonic_orders:
        if h <= 1:
            continue
        # Lower amplitude for higher harmonics.
        amp = fundamental_amplitude * (harmonic_relative_amplitude / float(h))
        y += amp * np.sin(float(h) * phi)

    if noise_std > 0.0:
        rng = np.random.default_rng(0)
        y += rng.normal(0.0, noise_std, size=y.shape)

    return np.asarray(y, dtype=dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PRB (perceptual rub & buzz) performance.")
    parser.add_argument("--wav", type=str, default="", help="Optional path to a mono/stereo WAV file.")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--duration", type=float, default=65.0, help="Seconds (use >60 for your test).")
    parser.add_argument("--start", type=float, default=80.0, help="Chirp start frequency (Hz).")
    parser.add_argument("--stop", type=float, default=8000.0, help="Chirp stop frequency (Hz).")
    parser.add_argument("--harmonics", type=str, default="2-35", help='e.g. "2-35" or "10,11,12"')
    parser.add_argument("--window", type=int, default=2048, help="STFT window size")
    parser.add_argument("--hop", type=int, default=1024, help="STFT hop size")
    parser.add_argument("--noise-std", type=float, default=0.0, help="White noise std-dev (signal units)")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile and print top stats")
    parser.add_argument(
        "--profile-out",
        type=str,
        default=os.path.join("log", "prb_profile.pstats"),
        help="pstats output path",
    )
    args = parser.parse_args()

    harmonic_orders = _parse_harmonic_orders(args.harmonics)

    if args.wav:
        file_sr, data = wavfile.read(args.wav)
        if int(file_sr) != int(args.sr):
            raise SystemExit(f"WAV sample_rate={file_sr} != --sr={args.sr}; please resample first.")
        x = np.asarray(data)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if np.issubdtype(x.dtype, np.integer):
            max_val = float(np.iinfo(x.dtype).max)
            x = x.astype(np.float32) / max_val
        else:
            x = x.astype(np.float32)
        recorded_signal = x
        args.duration = float(len(recorded_signal) / float(args.sr))
    else:
        recorded_signal = generate_chirp_with_harmonics(
            sample_rate=args.sr,
            duration_s=args.duration,
            start_hz=args.start,
            stop_hz=args.stop,
            harmonic_orders=harmonic_orders,
            noise_std=args.noise_std,
            dtype=np.float32,
        )

    stimulus_metadata = {
        "stimulus_method": "chirps",
        "stimulus_type": "linear",
        "start_freq": float(args.start),
        "stop_freq": float(args.stop),
        "total_time": float(args.duration),
        "repeat_times": 1,
        "sample_rate": int(args.sr),
    }

    analyzer = PerceptualChirpSignalHD(sample_rate=args.sr)

    def _run():
        return analyzer.compute_distortion(
            recorded_signal=recorded_signal,
            stimulus_metadata=stimulus_metadata,
            harmonic_orders=harmonic_orders,
            harmonic_mask=None,
            stft_window_size=args.window,
            stft_hop_size=args.hop,
            masking_config=None,
            spl_calibration_db=0.0,
            noise_spectrum=None,
        )

    t0 = time.perf_counter()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        result = _run()
        pr.disable()

        os.makedirs(os.path.dirname(os.path.abspath(args.profile_out)), exist_ok=True)
        pr.dump_stats(args.profile_out)
        s = io.StringIO()
        pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(40)
        print(s.getvalue())
    else:
        result = _run()
    t1 = time.perf_counter()

    n_frames = int(len(result.get("perceptual_loudness", [])))
    dt = t1 - t0
    print(
        f"PRB benchmark: sr={args.sr}Hz duration={args.duration:.2f}s "
        f"frames={n_frames} window={args.window} hop={args.hop} "
        f"harmonics={len(harmonic_orders)} runtime={dt:.3f}s "
        f"({(dt / max(n_frames, 1)):.6f}s/frame)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
