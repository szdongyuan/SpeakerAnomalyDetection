import argparse
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile

# Avoid Matplotlib writing to user home (often blocked / slow).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_PROJECT_ROOT, ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt

from base.pre_processing.psychoacoustic_utils import spl_to_phons
from base.soundcard_calibration_manager import get_mic_deviation_value


@dataclass(frozen=True)
class ToneResult:
    order: int
    freq_hz: float
    spl_db: float
    phons: float
    path: str


_FREQ_RE = re.compile(r"(?:^|/|\\)tone_(?P<freq>\d+(?:\.\d+)?)Hz_", re.IGNORECASE)


def _read_wav_mono(path: str) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        # Convert to float in [-1, 1] range for PCM.
        scale = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float64) / scale
    else:
        data = data.astype(np.float64)
    return int(sr), data


def _parse_freq_from_filename(path: str) -> float:
    m = _FREQ_RE.search(path)
    if not m:
        raise ValueError(f"Cannot parse frequency from filename: {path}")
    return float(m.group("freq"))


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot phons vs harmonic order for single-tone WAVs.")
    parser.add_argument("--dir", type=str, default="audio_data/out_wav_single", help="Directory of tone WAV files.")
    parser.add_argument("--f0", type=float, default=100.0, help="Fundamental frequency (Hz) used to map order=f/f0.")
    parser.add_argument(
        "--calibration-db",
        type=float,
        default=None,
        help="Calibration offset in dB (defaults to mic_calibration.txt if present, else 0).",
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=0.5,
        help="Seconds to trim from start and end before RMS (avoid transients).",
    )
    parser.add_argument("--out", type=str, default=os.path.join("log", "harmonic_phons.png"))
    parser.add_argument("--csv-out", type=str, default=os.path.join("log", "harmonic_phons.csv"))
    args = parser.parse_args()

    tones_dir = args.dir
    if not os.path.isdir(tones_dir):
        raise FileNotFoundError(f"Directory not found: {tones_dir}")

    calibration_db = args.calibration_db
    if calibration_db is None:
        calibration_db = float(get_mic_deviation_value() or 0.0)

    calibration_multiplier = float(np.power(10.0, calibration_db / 20.0)) if calibration_db != 0.0 else 1.0

    reference_pressure = 20e-6

    results: list[ToneResult] = []
    for name in os.listdir(tones_dir):
        if not name.lower().endswith(".wav"):
            continue
        path = os.path.join(tones_dir, name)
        freq_hz = _parse_freq_from_filename(path)
        order_f = freq_hz / float(args.f0)
        order = int(round(order_f))
        if not np.isclose(order_f, order, rtol=0.0, atol=1e-6):
            continue

        sr, y = _read_wav_mono(path)
        trim_n = int(max(args.trim, 0.0) * sr)
        if y.size > 2 * trim_n:
            y = y[trim_n:-trim_n]
        rms_val = _rms(y) * calibration_multiplier
        spl_db = 20.0 * np.log10(max(rms_val, 1e-30) / reference_pressure)
        phons = float(spl_to_phons(np.array([freq_hz], dtype=float), np.array([spl_db], dtype=float))[0])

        results.append(ToneResult(order=order, freq_hz=freq_hz, spl_db=float(spl_db), phons=phons, path=path))

    if not results:
        raise RuntimeError(f"No WAV files parsed in {tones_dir}")

    results.sort(key=lambda r: r.order)

    orders = np.array([r.order for r in results], dtype=int)
    phons = np.array([r.phons for r in results], dtype=float)

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(orders, phons, "-o", markersize=3, linewidth=1)
    plt.xlabel("Harmonic order (k)")
    plt.ylabel("Perceived loudness (phons)")
    plt.title(f"Single-tone loudness vs harmonic order (f0={args.f0:g} Hz, calibration={calibration_db:g} dB)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

    print(f"Wrote plot: {out_path}")
    csv_path = args.csv_out
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("order,freq_hz,spl_db,phons,path\n")
        for r in results:
            f.write(f"{r.order},{r.freq_hz:.6f},{r.spl_db:.6f},{r.phons:.6f},{r.path}\n")
    print(f"Wrote CSV: {csv_path}")
    print(f"Orders: {orders.min()}..{orders.max()} (n={len(orders)})")
    print(f"Phons: min={phons.min():.3f} max={phons.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
