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

from base.soundcard_calibration_manager import get_mic_deviation_value
from base.core_algorithm.psychoacoustic.psychoacoustic_utils import freq_to_bark

try:
    import mosqito
except Exception as e:  # pragma: no cover
    mosqito = None
    _MOSQITO_IMPORT_ERROR = e


@dataclass(frozen=True)
class ToneResult:
    order: int
    freq_hz: float
    spl_db: float
    sones_total: float
    phons_total: float
    specific_sones_per_bark_at_f: float
    specific_phon_equiv_at_f: float
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

def _sones_to_phons(sones: float) -> float:
    sones = float(sones)
    if not np.isfinite(sones) or sones <= 0.0:
        return 0.0
    if sones < 1.0:
        return float(40.0 * np.power(sones, 0.4))
    return float(40.0 + 10.0 * np.log2(sones))

def _sample_specific_loudness_sones_per_bark(
    n_specific: np.ndarray,
    bark_axis: np.ndarray,
    target_bark: float,
    *,
    edge_window_bark: float = 0.6,
) -> float:
    """
    Sample the specific loudness N'(z) in sones/bark at a target Bark location.

    For target Bark values outside the returned bark axis range (e.g., > 24 Bark),
    use a windowed mean near the nearest edge. This avoids a hard clamp to the
    last bin, which can be exactly zero and produce misleading '0' results.
    """
    n_specific = np.asarray(n_specific, dtype=np.float64).reshape(-1)
    bark_axis = np.asarray(bark_axis, dtype=np.float64).reshape(-1)
    if n_specific.size == 0 or bark_axis.size == 0 or n_specific.size != bark_axis.size:
        raise ValueError(f"Invalid specific loudness arrays: {n_specific.shape} vs {bark_axis.shape}")

    z0 = float(target_bark)
    bark_min = float(bark_axis[0])
    bark_max = float(bark_axis[-1])

    # Normal in-range sampling.
    if bark_min <= z0 <= bark_max:
        return float(np.interp(z0, bark_axis, n_specific))

    # Out-of-range: windowed mean near the closest edge.
    window = float(max(edge_window_bark, 0.0))
    if window <= 0.0:
        return float(n_specific[0] if z0 < bark_min else n_specific[-1])

    half = 0.5 * window
    if (bark_max - bark_min) <= window:
        return float(np.mean(n_specific))

    if z0 < bark_min:
        center = bark_min + half
    else:
        center = bark_max - half

    lo = center - half
    hi = center + half
    mask = (bark_axis >= lo) & (bark_axis <= hi)
    if not np.any(mask):
        return float(n_specific[0] if z0 < bark_min else n_specific[-1])
    return float(np.mean(n_specific[mask]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot phons vs harmonic order for single-tone WAVs (mosqito ISO 532-1 loudness)."
    )
    parser.add_argument("--dir", type=str, default="audio_data/out_wav_single", help="Directory of tone WAV files.")
    parser.add_argument("--f0", type=float, default=100.0, help="Fundamental frequency (Hz) used to map order=f/f0.")
    parser.add_argument("--min-order", type=int, default=2, help="Minimum harmonic order to include.")
    parser.add_argument("--max-order", type=int, default=None, help="Maximum harmonic order to include.")
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

    if mosqito is None:
        raise ImportError(
            "mosqito is required for this script, but it failed to import: "
            f"{type(_MOSQITO_IMPORT_ERROR).__name__}: {_MOSQITO_IMPORT_ERROR}"
        )

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
        if order < int(args.min_order):
            continue
        if args.max_order is not None and order > int(args.max_order):
            continue

        sr, y = _read_wav_mono(path)
        trim_n = int(max(args.trim, 0.0) * sr)
        if y.size > 2 * trim_n:
            y = y[trim_n:-trim_n]
        y_cal = y * calibration_multiplier

        rms_val = _rms(y_cal)
        spl_db = 20.0 * np.log10(max(rms_val, 1e-30) / reference_pressure)
        sones_total, n_spec, bark_axis = mosqito.loudness_zwst(y_cal, float(sr), field_type="free")
        sones_total = float(sones_total)
        phons_total = _sones_to_phons(sones_total)

        n_spec = np.asarray(n_spec, dtype=np.float64).reshape(-1)
        bark_axis = np.asarray(bark_axis, dtype=np.float64).reshape(-1)
        if n_spec.size != bark_axis.size or n_spec.size == 0:
            raise ValueError(f"mosqito returned invalid specific loudness for {path}: {n_spec.shape} vs {bark_axis.shape}")

        # "Spreaded loudness at this frequency": sample the specific loudness N'(z) at z=bark(freq).
        # Note: N' is in sones/bark (a density). For ecosystem reasons, we also provide a 'phon-equivalent'
        # by applying the standard sones->phons mapping to the sampled density value directly.
        z0 = float(freq_to_bark(np.array([freq_hz], dtype=np.float64))[0])
        specific_sones_per_bark_at_f = _sample_specific_loudness_sones_per_bark(n_spec, bark_axis, z0)
        specific_phon_equiv_at_f = _sones_to_phons(specific_sones_per_bark_at_f)

        results.append(
            ToneResult(
                order=order,
                freq_hz=freq_hz,
                spl_db=float(spl_db),
                sones_total=sones_total,
                phons_total=phons_total,
                specific_sones_per_bark_at_f=specific_sones_per_bark_at_f,
                specific_phon_equiv_at_f=specific_phon_equiv_at_f,
                path=path,
            )
        )

    if not results:
        raise RuntimeError(f"No WAV files parsed in {tones_dir}")

    results.sort(key=lambda r: r.order)

    orders = np.array([r.order for r in results], dtype=int)
    phons_total = np.array([r.phons_total for r in results], dtype=float)
    phons_specific = np.array([r.specific_phon_equiv_at_f for r in results], dtype=float)

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(orders, phons_total, "-o", markersize=3, linewidth=1, label="Total loudness (mosqito N → phon)")
    plt.plot(
        orders,
        phons_specific,
        "-o",
        markersize=3,
        linewidth=1,
        label="Specific loudness at f (mosqito N'(bark(f)) → phon-equiv)",
    )
    plt.xlabel("Harmonic order (k)")
    plt.ylabel("Perceived loudness (phons)")
    plt.title(f"Single-tone loudness vs harmonic order (f0={args.f0:g} Hz, calibration={calibration_db:g} dB)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

    print(f"Wrote plot: {out_path}")
    csv_path = args.csv_out
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "order,freq_hz,spl_db,sones_total,phons_total,specific_sones_per_bark_at_f,specific_phon_equiv_at_f,path\n"
        )
        for r in results:
            f.write(
                f"{r.order},{r.freq_hz:.6f},{r.spl_db:.6f},"
                f"{r.sones_total:.6f},{r.phons_total:.6f},"
                f"{r.specific_sones_per_bark_at_f:.9f},{r.specific_phon_equiv_at_f:.6f},"
                f"{r.path}\n"
            )
    print(f"Wrote CSV: {csv_path}")
    print(f"Orders: {orders.min()}..{orders.max()} (n={len(orders)})")
    print(f"Total phons: min={phons_total.min():.3f} max={phons_total.max():.3f}")
    print(f"Specific phon-equiv: min={phons_specific.min():.3f} max={phons_specific.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
