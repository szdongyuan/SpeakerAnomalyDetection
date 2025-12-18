import argparse
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_PROJECT_ROOT, ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt

from base.core_algorithm.perceptual_rubbuzz_sc.peaq_loudness_model import PEAQLoudnessConfig, PEAQLoudnessModel


@dataclass(frozen=True)
class ToneRow:
    frequency_hz: float
    pa_per_v: float
    rms_pa: float
    spl_db: float
    n_total_phons: float
    totalnl_phons: float
    nl_at_f_phon_equiv: float
    file: str


_FREQ_RE = re.compile(r"(?:^|/|\\)tone_(?P<freq>\d+(?:\.\d+)?)Hz_", re.IGNORECASE)


def _read_wav_mono(path: str) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
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
    parser = argparse.ArgumentParser(description="Plot Listen-paper simplified PEAQ SC loudness for tone WAVs.")
    parser.add_argument("--dir", type=str, default="audio_data/out_wav_single", help="Directory of tone WAV files.")
    parser.add_argument("--pa-per-v", type=float, default=0.1, help="Calibration: Pa = V * pa_per_v.")
    parser.add_argument(
        "--ear-term3-exp",
        type=float,
        default=None,
        help="Override Eq.1 high-frequency roll-off exponent (paper: 3.6).",
    )
    parser.add_argument(
        "--ear-term3-coeff",
        type=float,
        default=None,
        help="Override Eq.1 high-frequency roll-off coefficient (paper: -1e-3).",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=0,
        help="FFT size for the spectrum axis (0 means use sample_rate -> 1 Hz bins).",
    )
    parser.add_argument("--trim", type=float, default=0.5, help="Seconds to trim from start/end before RMS.")
    parser.add_argument("--out", type=str, default=os.path.join("log", "peaq_sc_tone_curve.png"))
    parser.add_argument("--csv-out", type=str, default=os.path.join("log", "peaq_sc_tone_curve.csv"))
    args = parser.parse_args()

    tones_dir = args.dir
    if not os.path.isdir(tones_dir):
        raise FileNotFoundError(f"Directory not found: {tones_dir}")

    reference_pressure_pa = 20e-6

    rows: list[ToneRow] = []
    model: PEAQLoudnessModel | None = None
    freqs_hz: np.ndarray | None = None
    config: PEAQLoudnessConfig | None = None

    cfg_kwargs: dict[str, float] = {}
    if args.ear_term3_exp is not None:
        cfg_kwargs["ear_weighting_term3_exponent"] = float(args.ear_term3_exp)
    if args.ear_term3_coeff is not None:
        cfg_kwargs["ear_weighting_term3_coeff"] = float(args.ear_term3_coeff)
    config = PEAQLoudnessConfig(**cfg_kwargs)

    for name in sorted(os.listdir(tones_dir)):
        if not name.lower().endswith(".wav"):
            continue

        path = os.path.join(tones_dir, name)
        freq_hz = _parse_freq_from_filename(path)
        sr, y_v = _read_wav_mono(path)

        trim_n = int(max(args.trim, 0.0) * sr)
        if y_v.size > 2 * trim_n:
            y_v = y_v[trim_n:-trim_n]

        y_pa = y_v * float(args.pa_per_v)
        rms_pa = _rms(y_pa)
        spl_db = 20.0 * np.log10(max(rms_pa, 1e-30) / reference_pressure_pa)

        n_fft = int(args.n_fft) if int(args.n_fft) > 0 else int(sr)
        if model is None or freqs_hz is None or freqs_hz.size != (n_fft // 2 + 1):
            freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            model = PEAQLoudnessModel(freqs_hz, config=config)

        # Paper-style spectrum: single RMS amplitude at the closest rFFT bin.
        spec_test = np.zeros(freqs_hz.size, dtype=np.float64)
        idx = int(np.argmin(np.abs(freqs_hz - freq_hz)))
        spec_test[idx] = rms_pa

        # For pure-tone debugging, treat the stimulus/f0 reference as silence so internal noise cancels.
        spec_ref = np.zeros_like(spec_test)

        out_n = model.compute_loudness_from_single_spectrum(spec_test)
        out_nl = model.compute_partial_loudness_from_spectra(spec_test, spec_ref, apply_level_adaptation=False)

        nl_at_f = float(out_nl.interpolate_specific_phons_equiv(np.array([freq_hz], dtype=np.float64))[0, 0])

        rows.append(
            ToneRow(
                frequency_hz=float(freq_hz),
                pa_per_v=float(args.pa_per_v),
                rms_pa=float(rms_pa),
                spl_db=float(spl_db),
                n_total_phons=float(out_n.n_total_phons[0]),
                totalnl_phons=float(out_nl.n_total_phons[0]),
                nl_at_f_phon_equiv=nl_at_f,
                file=name,
            )
        )

    if not rows:
        raise RuntimeError(f"No WAV files parsed in {tones_dir}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    csv_dir = os.path.dirname(args.csv_out)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    rows.sort(key=lambda r: r.frequency_hz)
    f_hz = np.array([r.frequency_hz for r in rows], dtype=np.float64)
    n_total_phons = np.array([r.n_total_phons for r in rows], dtype=np.float64)
    totalnl_phons = np.array([r.totalnl_phons for r in rows], dtype=np.float64)
    nl_at_f_phons = np.array([r.nl_at_f_phon_equiv for r in rows], dtype=np.float64)

    plt.figure(figsize=(14, 6))
    plt.plot(f_hz, n_total_phons, "-o", markersize=2, linewidth=1, label="N_total (Eq.12) → phons")
    plt.plot(f_hz, totalnl_phons, "-o", markersize=2, linewidth=1, label="TotalNL (Eq.15, ref=silence) → phons")
    plt.plot(
        f_hz,
        nl_at_f_phons,
        "-o",
        markersize=2,
        linewidth=1,
        label="NL at f (interpolated, phon-equiv)",
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Loudness (phons / phon-equiv)")
    title = f"PEAQ SC tone loudness curve (pa_per_v={float(args.pa_per_v):g}, n_fft={n_fft})"
    if args.ear_term3_exp is not None or args.ear_term3_coeff is not None:
        title += (
            f"\nEq.1 term3: coeff={config.ear_weighting_term3_coeff:g}, exp={config.ear_weighting_term3_exponent:g}"
        )
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote plot: {args.out}")

    with open(args.csv_out, "w", encoding="utf-8") as f:
        f.write(
            "frequency_hz,pa_per_v,rms_pa,spl_db,n_total_phons,totalnl_phons,nl_at_f_phon_equiv,file\n"
        )
        for r in rows:
            f.write(
                f"{r.frequency_hz:.6f},{r.pa_per_v:.6g},{r.rms_pa:.12g},{r.spl_db:.6f},"
                f"{r.n_total_phons:.6f},{r.totalnl_phons:.6f},{r.nl_at_f_phon_equiv:.6f},{r.file}\n"
            )
    print(f"Wrote CSV: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
