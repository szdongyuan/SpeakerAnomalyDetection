"""Prominence Ratio (突出比, PR) 算法编排层。

信号预处理 → FFT/Welch PSD → 计权 → 临界带功率积分 → PR 频谱
→ 候选主音检测/BPF/同带分组 → ECMA prominent + 客户 4/2/4 判定。

ECMA 临界带公式统一由 ecma_critical_band 模块提供。
PR = 10·lg(XM / ((XL + XU) / 2))，线性功率域。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy import signal as sp_signal

from base.core_algorithm.response import ecma_critical_band as ecb

P_REF_PA = 20e-6
MAIN_TONE_MIN_GAP_DB = 3.0
CANDIDATE_PEAK_PROMINENCE_DB = MAIN_TONE_MIN_GAP_DB


@dataclass
class ProminenceRatioParams:
    """PR 分析参数。"""

    f_min: float = 100.0
    f_max: float = 20000.0
    standard: str = "ecma74_annexD"
    critical_band_mode: str = "ecma"
    mode_profile: str = "ecma2022"
    ecma_ft_min_hz: float = ecb.ECMA_FT_MIN_HZ
    ecma_scope_ft_max_hz: float = ecb.ECMA_SCOPE_FT_MAX_HZ
    ecma_formula_fit_max_hz: float = ecb.ECMA_FORMULA_FIT_MAX_HZ
    window_samples: int = 65536
    target_resolution_hz: float = 1.0
    overlap_ratio: float = 0.75
    window: str = "hann"
    weighting: str = "auto"
    customer_band_ratio: float = 0.15
    dc_removal: str = "mean"
    highpass_hz: float = 20.0
    bpf_enabled: bool = False
    blade_count: int = 0
    rpm: int = 0
    bpf_tolerance_percent: float = 5.0
    include_harmonics_in_customer_judgement: bool = False
    user_tone_frequencies: tuple[float, ...] = ()

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> tuple["ProminenceRatioParams", list[str]]:
        """从 UI/JSON 配置构造参数，返回 (params, config_warnings)。"""
        cfg = cfg or {}
        adv = cfg.get("advanced", {}) or {}
        warnings: list[str] = []

        def pick(key, default):
            if key in adv:
                return adv[key]
            if key in cfg:
                return cfg[key]
            return default

        defaults = cls()
        mode = str(pick("critical_band_mode", defaults.critical_band_mode))
        raw_weighting = str(pick("weighting", defaults.weighting))
        effective_weighting, w_warn = ecb.resolve_weighting(mode, raw_weighting)
        warnings.extend(w_warn)

        params = cls(
            f_min=float(pick("f_min", defaults.f_min)),
            f_max=float(pick("f_max", defaults.f_max)),
            standard=str(pick("standard", defaults.standard)),
            critical_band_mode=mode,
            mode_profile=str(pick("mode_profile", defaults.mode_profile)),
            window_samples=int(pick("window_samples", defaults.window_samples)),
            target_resolution_hz=float(pick("target_resolution_hz", defaults.target_resolution_hz)),
            overlap_ratio=float(pick("overlap_ratio", defaults.overlap_ratio)),
            window=str(pick("window", defaults.window)),
            weighting=effective_weighting,
            customer_band_ratio=float(pick("customer_band_ratio", defaults.customer_band_ratio)),
            dc_removal=str(pick("dc_removal", defaults.dc_removal)),
            highpass_hz=float(pick("highpass_hz", defaults.highpass_hz)),
            bpf_enabled=bool(pick("bpf_enabled", defaults.bpf_enabled)),
            blade_count=int(pick("blade_count", defaults.blade_count)),
            rpm=int(pick("rpm", defaults.rpm)),
            bpf_tolerance_percent=float(pick("bpf_tolerance_percent", defaults.bpf_tolerance_percent)),
            include_harmonics_in_customer_judgement=bool(
                pick("include_harmonics_in_customer_judgement", defaults.include_harmonics_in_customer_judgement)
            ),
            user_tone_frequencies=cls._parse_user_tone_frequencies(pick("user_tone_frequencies", "")),
        )
        return params, warnings

    @staticmethod
    def _parse_user_tone_frequencies(raw) -> tuple[float, ...]:
        if not raw:
            return ()
        if isinstance(raw, (list, tuple)):
            vals = raw
        else:
            vals = str(raw).replace(";", ",").replace(" ", ",").split(",")
        result = []
        for v in vals:
            try:
                f = float(v)
                if np.isfinite(f) and f > 0:
                    result.append(f)
            except (ValueError, TypeError):
                continue
        return tuple(sorted(set(result)))


@dataclass
class ProminenceToneResult:
    frequency_hz: float
    peak_db: float
    target_band_hz: tuple[float, float]
    lower_adjacent_band_hz: tuple[float, float]
    upper_adjacent_band_hz: tuple[float, float]
    target_power_db: float
    lower_adjacent_power_db: float
    upper_adjacent_power_db: float
    adjacent_mean_power_db: float
    pr_db: float
    ecma_prominence_limit_db: Optional[float]
    ecma_prominent: Optional[bool]
    limit_db: float
    customer_ok: Optional[bool]
    margin_db: float
    is_ok: bool
    same_band_group_id: Optional[int]
    same_band_representative: bool
    harmonic_order: Optional[int]
    valid_main_tone: bool
    bpf_verified: Optional[bool]
    user_specified: bool = False
    invalid_reasons: list[str] = field(default_factory=list)


@dataclass
class ProminenceRatioSpectrumResult:
    frequency_hz: np.ndarray
    band_power_db: np.ndarray
    pr_db: np.ndarray
    fft_magnitude_db: np.ndarray
    fft_freq_hz: np.ndarray
    main_tones: list[ProminenceToneResult]
    max_pr_db: float
    max_pr_frequency_hz: float
    decision_status: str
    overall_ok: Optional[bool]
    max_exceed_db: float
    no_valid_main_tone: bool
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)



def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return int(2 ** np.ceil(np.log2(n)))


def _customer_limit_db(ft: float, fan_pr_limits: list) -> float:
    for band in fan_pr_limits:
        f_lo, f_hi, limit = band[0], band[1], band[2]
        if f_lo <= ft < f_hi:
            return float(limit)
    return 4.0


def _power_to_level_db(power: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    arr = np.asarray(power, dtype=float)
    level = 10.0 * np.log10(np.maximum(arr, 1e-30) / (P_REF_PA ** 2))
    if np.ndim(power) == 0:
        return float(level)
    return level


def _empty_invalid_result(reason: str, warnings: list[str], metadata: dict) -> ProminenceRatioSpectrumResult:
    empty = np.asarray([], dtype=float)
    return ProminenceRatioSpectrumResult(
        frequency_hz=empty,
        band_power_db=empty,
        pr_db=empty,
        fft_magnitude_db=empty,
        fft_freq_hz=empty,
        main_tones=[],
        max_pr_db=float("nan"),
        max_pr_frequency_hz=float("nan"),
        decision_status="invalid",
        overall_ok=None,
        max_exceed_db=0.0,
        no_valid_main_tone=True,
        warnings=[*warnings, reason],
        metadata=metadata,
    )



class ProminenceRatioAnalyzer:

    MIN_SAMPLES = 1024

    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)

    def _preprocess(self, signal: np.ndarray, v2pa_factor: float, params: ProminenceRatioParams,
                    warnings: list[str]) -> np.ndarray:
        x = np.asarray(signal, dtype=float).flatten() * float(v2pa_factor)
        if params.dc_removal == "mean":
            x = x - np.mean(x)
        if params.highpass_hz and params.highpass_hz > 0.0:
            nyq = self.sample_rate / 2.0
            wc = params.highpass_hz / nyq
            if 0.0 < wc < 1.0:
                b, a = sp_signal.butter(2, wc, btype="highpass")
                x = sp_signal.filtfilt(b, a, x)
            else:
                warnings.append(f"高通截频 {params.highpass_hz}Hz 超出有效范围，已跳过")
        return x

    def _compute_psd(self, x: np.ndarray, params: ProminenceRatioParams,
                     warnings: list[str]) -> tuple[np.ndarray, np.ndarray]:
        nperseg = int(params.window_samples)
        target_grid = _next_power_of_two(int(self.sample_rate / max(params.target_resolution_hz, 1e-6)))
        nfft = max(nperseg, target_grid)
        noverlap = int(nperseg * params.overlap_ratio)
        freq, psd = sp_signal.welch(
            x,
            fs=self.sample_rate,
            window=params.window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            scaling="density",
        )
        return np.asarray(freq, dtype=float), np.asarray(psd, dtype=float)

    @staticmethod
    def _build_cumulative_power(freq: np.ndarray, psd: np.ndarray) -> np.ndarray:
        seg = 0.5 * (psd[1:] + psd[:-1]) * np.diff(freq)
        return np.concatenate(([0.0], np.cumsum(seg)))

    @staticmethod
    def _band_power(edges_lo: np.ndarray, edges_hi: np.ndarray,
                    freq: np.ndarray, cum: np.ndarray) -> np.ndarray:
        lo = np.interp(edges_lo, freq, cum)
        hi = np.interp(edges_hi, freq, cum)
        return np.maximum(hi - lo, 0.0)

    @staticmethod
    def _band_edges(f: np.ndarray, params: ProminenceRatioParams):
        """对频率数组 f 向量化计算 (f1l,f2l,f1m,f2m,f1u,f2u,corr)。"""
        f = np.asarray(f, dtype=float)
        if params.critical_band_mode == "customer_15pct":
            bw = params.customer_band_ratio * f
            half = bw / 2.0
            f1m, f2m = f - half, f + half
            f1l, f2l = f1m - bw, f1m
            f1u, f2u = f2m, f2m + bw
            corr = np.ones_like(f)
        else:
            ft = f
            dfc = ecb.critical_bandwidth_hz(ft)
            half = dfc / 2.0
            f1m_arith, f2m_arith = ft - half, ft + half
            f1m_geo = (-dfc + np.sqrt(dfc * dfc + 4.0 * ft * ft)) / 2.0
            f2m_geo = f1m_geo + dfc
            is_arith = ft <= ecb.MIDDLE_BAND_ARITH_MAX_HZ
            f1m = np.where(is_arith, f1m_arith, f1m_geo)
            f2m = np.where(is_arith, f2m_arith, f2m_geo)
            f2l = f1m
            f1u = f2m
            f1l = np.where(
                ft < ecb.LOW_FREQ_CORRECTION_FT_HZ,
                20.0,
                np.where(ft <= 1600.0,
                         -149.5 + 1.001 * ft - 6.90e-5 * ft * ft,
                         6.8 + 0.806 * ft - 8.20e-6 * ft * ft),
            )
            f2u = np.where(ft <= 1600.0,
                           149.5 + 1.035 * ft + 7.70e-5 * ft * ft,
                           3.3 + 1.215 * ft + 2.16e-5 * ft * ft)
            dfl = np.maximum(f2l - f1l, 1e-9)
            corr = np.where(ft <= ecb.LOW_FREQ_CORRECTION_FT_HZ, 100.0 / dfl, 1.0)
        return f1l, f2l, f1m, f2m, f1u, f2u, corr

    @staticmethod
    def _analysis_f_max(params: ProminenceRatioParams, nyq: float) -> float:
        f_max = min(float(params.f_max), float(nyq))
        if params.critical_band_mode != "customer_15pct":
            f_max = min(f_max, float(params.ecma_scope_ft_max_hz))
        return f_max

    def _pr_estimate(self, f: np.ndarray, freq: np.ndarray, cum: np.ndarray,
                     params: ProminenceRatioParams, nyq: float) -> np.ndarray:
        f1l, f2l, f1m, f2m, f1u, f2u, corr = self._band_edges(f, params)
        xm = self._band_power(f1m, f2m, freq, cum)
        xl = self._band_power(f1l, f2l, freq, cum) * corr
        xu = self._band_power(f1u, f2u, freq, cum)
        adj = (xl + xu) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = 10.0 * np.log10(np.where(adj > 0, xm / adj, np.nan))
        invalid = (f1l <= 0.0) | (f2u > nyq)
        if params.critical_band_mode != "customer_15pct":
            invalid = invalid | (f < params.ecma_ft_min_hz) | (f > params.ecma_scope_ft_max_hz)
        return np.where(invalid, np.nan, pr)

    def _compute_pr_spectrum(self, freq: np.ndarray, cum: np.ndarray,
                             params: ProminenceRatioParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nyq = self.sample_rate / 2.0
        f_max = self._analysis_f_max(params, nyq)
        mask = (freq >= params.f_min) & (freq <= f_max)
        f = freq[mask]
        if f.size == 0:
            return f, f, f
        _f1l, _f2l, f1m, f2m, _f1u, _f2u, _corr = self._band_edges(f, params)
        target_power = self._band_power(f1m, f2m, freq, cum)
        invalid = (f1m <= 0.0) | (f2m > nyq)
        band_power_db = _power_to_level_db(target_power)
        band_power_db = np.where(invalid, np.nan, band_power_db)
        pr = self._pr_estimate(f, freq, cum, params, nyq)
        return f, band_power_db, pr

    def _detect_tones(self, freq: np.ndarray, psd_w: np.ndarray, cum: np.ndarray,
                      params: ProminenceRatioParams, fan_pr_limits: list,
                      warnings: list[str]) -> list[ProminenceToneResult]:
        nyq = self.sample_rate / 2.0
        mag_db = 10.0 * np.log10(np.maximum(psd_w, 1e-30))
        f_max = self._analysis_f_max(params, nyq)
        band_mask = (freq >= params.f_min) & (freq <= f_max)
        mag_in = mag_db[band_mask]
        freq_in = freq[band_mask]
        if freq_in.size < 3:
            return []

        peaks, _props = sp_signal.find_peaks(mag_in, prominence=CANDIDATE_PEAK_PROMINENCE_DB)

        user_specified_bins: set[int] = set()
        if params.user_tone_frequencies:
            for uf in params.user_tone_frequencies:
                if freq_in[0] <= uf <= freq_in[-1]:
                    idx = int(np.argmin(np.abs(freq_in - uf)))
                    user_specified_bins.add(idx)
            all_peaks = sorted(set(peaks.tolist()) | user_specified_bins)
            peaks = np.array(all_peaks, dtype=int)

        bpf_hz = (params.blade_count * params.rpm / 60.0) if (params.bpf_enabled and params.rpm > 0) else None

        tones: list[ProminenceToneResult] = []
        for pk in peaks:
            ft = float(freq_in[pk])
            peak_db = float(mag_in[pk])
            invalid_reasons: list[str] = []

            triplet = ecb.get_band_triplet(
                ft,
                mode=params.critical_band_mode,
                customer_band_ratio=params.customer_band_ratio,
                ft_min_hz=params.ecma_ft_min_hz,
                ft_max_hz=params.ecma_scope_ft_max_hz,
                fit_max_hz=params.ecma_formula_fit_max_hz,
                nyquist_hz=nyq,
            )
            if not triplet.valid:
                invalid_reasons.append(triplet.reason or "invalid band")

            (f1l, f2l) = triplet.lower
            (f1m, f2m) = triplet.middle
            (f1u, f2u) = triplet.upper
            corr = ecb.low_freq_xl_correction_factor(ft, f1l, f2l) if params.critical_band_mode != "customer_15pct" else 1.0

            xm = float(self._band_power(np.array([f1m]), np.array([f2m]), freq, cum)[0])
            xl = float(self._band_power(np.array([f1l]), np.array([f2l]), freq, cum)[0]) * corr
            xu = float(self._band_power(np.array([f1u]), np.array([f2u]), freq, cum)[0])
            adj = (xl + xu) / 2.0
            if adj > 0 and xm > 0:
                pr_db = 10.0 * np.log10(xm / adj)
            else:
                pr_db = float("nan")
                invalid_reasons.append("non-positive band power")

            if params.critical_band_mode == "customer_15pct":
                ecma_limit = None
                ecma_prominent = None
            else:
                ecma_limit = ecb.ecma_prominence_limit_db(ft, params.ecma_scope_ft_max_hz)
                ecma_prominent = (pr_db >= ecma_limit) if (ecma_limit is not None and not np.isnan(pr_db)) else None

            limit_db = _customer_limit_db(ft, fan_pr_limits)
            if np.isnan(pr_db):
                customer_ok = None
                margin_db = float("nan")
            else:
                customer_ok = pr_db <= limit_db
                margin_db = limit_db - pr_db

            is_user_specified = (pk in user_specified_bins)
            valid_main_tone = is_user_specified or (len(invalid_reasons) == 0)

            bpf_verified: Optional[bool] = None
            harmonic_order: Optional[int] = None
            if bpf_hz:
                tol = params.bpf_tolerance_percent / 100.0
                ratio = ft / bpf_hz
                nearest = round(ratio)
                if nearest >= 1 and abs(ratio - nearest) <= tol:
                    bpf_verified = (nearest == 1)
                    harmonic_order = int(nearest) if nearest > 1 else None
                else:
                    bpf_verified = False
                if bpf_verified is False and not is_user_specified:
                    valid_main_tone = False
                    reason = "BPF/mechanical match not verified"
                    if reason not in invalid_reasons:
                        invalid_reasons.append(reason)

            tones.append(ProminenceToneResult(
                frequency_hz=ft,
                peak_db=peak_db,
                target_band_hz=(f1m, f2m),
                lower_adjacent_band_hz=(f1l, f2l),
                upper_adjacent_band_hz=(f1u, f2u),
                target_power_db=_power_to_level_db(xm) if xm > 0 else float("nan"),
                lower_adjacent_power_db=_power_to_level_db(xl) if xl > 0 else float("nan"),
                upper_adjacent_power_db=_power_to_level_db(xu) if xu > 0 else float("nan"),
                adjacent_mean_power_db=_power_to_level_db(adj) if adj > 0 else float("nan"),
                pr_db=pr_db,
                ecma_prominence_limit_db=ecma_limit,
                ecma_prominent=ecma_prominent,
                limit_db=limit_db,
                customer_ok=customer_ok,
                margin_db=margin_db,
                is_ok=bool(customer_ok) if customer_ok is not None else True,
                same_band_group_id=None,
                same_band_representative=True,
                harmonic_order=harmonic_order,
                valid_main_tone=valid_main_tone,
                bpf_verified=bpf_verified,
                user_specified=(pk in user_specified_bins),
                invalid_reasons=invalid_reasons,
            ))

        self._apply_frequency_interval_main_tone_rule(tones, fan_pr_limits)
        self._assign_same_band_groups(tones)
        return tones

    @staticmethod
    def _apply_frequency_interval_main_tone_rule(
        tones: list[ProminenceToneResult],
        fan_pr_limits: list,
    ) -> None:
        """按频段选出有效主音：段内最高峰且与次高峰差 >= 3dB。"""
        valid_candidates = [t for t in tones if t.valid_main_tone]
        if not valid_candidates:
            return

        intervals: list[tuple[float, float]] = []
        for band in fan_pr_limits or []:
            try:
                f_lo, f_hi = float(band[0]), float(band[1])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_hi > f_lo:
                intervals.append((f_lo, f_hi))
        if not intervals:
            intervals = [(min(t.frequency_hz for t in valid_candidates), max(t.frequency_hz for t in valid_candidates) + 1.0)]

        selected_ids: set[int] = set()
        for band_idx, (f_lo, f_hi) in enumerate(intervals):
            in_band = [
                t for t in valid_candidates
                if (f_lo <= t.frequency_hz < f_hi) or (band_idx == len(intervals) - 1 and f_lo <= t.frequency_hz <= f_hi)
            ]
            if not in_band:
                continue
            ranked = sorted(in_band, key=lambda t: t.peak_db, reverse=True)
            top = ranked[0]
            if len(ranked) > 1:
                non_user = [t for t in ranked if not t.user_specified]
                second = ranked[1]
                diff = top.peak_db - second.peak_db
                if diff < MAIN_TONE_MIN_GAP_DB:
                    reason = f"interval top-second gap {diff:.1f}dB < {MAIN_TONE_MIN_GAP_DB:.1f}dB"
                    for t in ranked:
                        if t.user_specified:
                            selected_ids.add(id(t))
                            continue
                        t.valid_main_tone = False
                        if reason not in t.invalid_reasons:
                            t.invalid_reasons.append(reason)
                    continue
            selected_ids.add(id(top))
            for t in ranked[1:]:
                if t.user_specified:
                    selected_ids.add(id(t))
                    continue
                t.valid_main_tone = False
                reason = "not highest tone in frequency interval"
                if reason not in t.invalid_reasons:
                    t.invalid_reasons.append(reason)

        for t in valid_candidates:
            if t.user_specified:
                selected_ids.add(id(t))
                continue
            if id(t) not in selected_ids and t.valid_main_tone:
                t.valid_main_tone = False
                reason = "outside configured PR frequency intervals"
                if reason not in t.invalid_reasons:
                    t.invalid_reasons.append(reason)

    @staticmethod
    def _assign_same_band_groups(tones: list[ProminenceToneResult]) -> None:
        """同一 middle critical band 内的多个有效主音归为一组，组内 PR 最大者为代表。

        采用 middle band 区间重叠的传递性归并（对频率排序后线性扫描）：两音的
        middle band 区间相交即并入同一组，避免“仅按首音判定”导致的非对称裂组。
        代表峰 same_band_representative=True 参与 OK/NG 汇总，其余仅展示，避免共享
        XM 总功率导致重复判 fail（见规划 4.1.4 同临界带多峰处理）。
        """
        for t in tones:
            t.same_band_group_id = None
            t.same_band_representative = bool(t.valid_main_tone)

        valid = sorted(
            (t for t in tones if t.valid_main_tone),
            key=lambda t: t.frequency_hz,
        )

        def _flush(members: list[ProminenceToneResult]) -> None:
            if not members:
                return
            rep = max(members, key=lambda m: (m.pr_db if not np.isnan(m.pr_db) else -np.inf))
            for m in members:
                m.same_band_representative = (m is rep)

        group_id = -1
        cur_upper = -np.inf
        cur_members: list[ProminenceToneResult] = []
        for t in valid:
            f1m, f2m = t.target_band_hz
            if cur_members and f1m <= cur_upper:
                cur_members.append(t)
                cur_upper = max(cur_upper, f2m)
            else:
                _flush(cur_members)
                group_id += 1
                cur_members = [t]
                cur_upper = f2m
            t.same_band_group_id = group_id
        _flush(cur_members)

    # -- 决策汇总 --------------------------------------------------------
    @staticmethod
    def _decide(tones: list[ProminenceToneResult],
                params: ProminenceRatioParams) -> tuple[str, Optional[bool], float, bool]:
        valid = [t for t in tones if t.valid_main_tone]
        if not valid:
            return "not_applicable", None, 0.0, True

        reps = [t for t in valid if t.same_band_representative]
        if not params.include_harmonics_in_customer_judgement:
            reps = [t for t in reps if t.harmonic_order is None]
        if not reps:
            return "not_applicable", None, 0.0, True

        if any(t.customer_ok is None for t in reps):
            return "invalid", None, 0.0, False

        exceed = [max(t.pr_db - t.limit_db, 0.0) for t in reps]
        max_exceed = max(exceed) if exceed else 0.0
        all_ok = all(t.customer_ok for t in reps)
        status = "pass" if all_ok else "fail"
        return status, all_ok, max_exceed, False

    # -- 主入口 ----------------------------------------------------------
    def compute(
        self,
        signal: np.ndarray,
        v2pa_factor: float = 1.0,
        params: Optional[ProminenceRatioParams] = None,
        fan_pr_limits: Optional[list] = None,
        initial_warnings: Optional[list[str]] = None,
    ) -> ProminenceRatioSpectrumResult:
        params = params or ProminenceRatioParams()
        fan_pr_limits = fan_pr_limits or [[100, 2000, 4], [2000, 5000, 2], [5000, 20000, 4]]
        warnings: list[str] = list(initial_warnings or [])

        # 计权兜底：即便 params 由直接构造（weighting 仍为 "auto"），
        # 也保证 customer_15pct 与 ecma 的默认计权一致，均为线性(Z)。
        effective_weighting, w_warn = ecb.resolve_weighting(params.critical_band_mode, params.weighting)
        warnings.extend(w_warn)

        metadata = {
            "standard": params.standard,
            "critical_band_mode": params.critical_band_mode,
            "mode_profile": params.mode_profile,
            "effective_weighting": effective_weighting,
            "sample_rate_hz": self.sample_rate,
            "window": params.window,
            "window_samples": params.window_samples,
            "effective_window_samples": params.window_samples,
            "calculation_downsampling": False,
            "overlap_ratio": params.overlap_ratio,
            "f_min": params.f_min,
            "f_max": params.f_max,
            "band_power_unit": "dB",
            "band_power_reference_pa": P_REF_PA,
        }

        x = np.asarray(signal, dtype=float).flatten()
        if x.size < self.MIN_SAMPLES:
            return _empty_invalid_result(f"信号过短（{x.size} < {self.MIN_SAMPLES}）", warnings, metadata)
        if params.window_samples <= 0:
            return _empty_invalid_result(f"窗口长度无效（window_samples={params.window_samples}）", warnings, metadata)
        if x.size < int(params.window_samples):
            return _empty_invalid_result(
                f"信号长度 {x.size} 小于配置窗口长度 {int(params.window_samples)}，PR 未计算；请延长采样时长或调小窗口长度",
                warnings,
                metadata,
            )

        nyq = self.sample_rate / 2.0
        if nyq < params.f_max:
            warnings.append(f"采样率不足以覆盖 {params.f_max}Hz，分析上限降为 {nyq}Hz")

        x = self._preprocess(signal, v2pa_factor, params, warnings)
        freq, psd = self._compute_psd(x, params, warnings)

        # 默认线性(Z)功率；显式 A/C 仅作为高级覆盖。
        psd_w = ecb.apply_weighting_to_psd(psd, freq, effective_weighting)
        metadata["effective_nfft"] = int((len(freq) - 1) * 2)

        cum = self._build_cumulative_power(freq, psd_w)
        f_axis, band_power_db, pr_spectrum = self._compute_pr_spectrum(freq, cum, params)
        tones = self._detect_tones(freq, psd_w, cum, params, fan_pr_limits, warnings)
        pr_valid_mask = np.isfinite(pr_spectrum)
        if pr_valid_mask.any():
            effective_pr_f_max = float(np.nanmax(f_axis[pr_valid_mask]))
        else:
            effective_pr_f_max = float("nan")
        configured_f_hi = self._analysis_f_max(params, nyq)
        metadata["effective_pr_f_max_hz"] = effective_pr_f_max
        pr_frequency_coverage_complete = True
        if np.isfinite(effective_pr_f_max) and effective_pr_f_max + max(params.target_resolution_hz, 1.0) < configured_f_hi:
            pr_frequency_coverage_complete = False
            f1l_req, f2l_req, f1m_req, f2m_req, f1u_req, f2u_req, _ = self._band_edges(
                np.array([configured_f_hi], dtype=float),
                params,
            )
            required_sr = 2.0 * float(f2u_req[0])
            metadata["required_sample_rate_for_configured_f_max_hz"] = required_sr
            warnings.append(
                f"PR 有效上限约为 {effective_pr_f_max:.0f}Hz；"
                f"{configured_f_hi:.0f}Hz 的右相邻临界带上边界需采样率 ≥ {required_sr:.0f}Hz，"
                f"当前采样率 {self.sample_rate}Hz 不足，超出部分无 PR 值"
            )
        metadata["pr_frequency_coverage_complete"] = pr_frequency_coverage_complete
        if freq.size > 1:
            df = float(np.median(np.diff(freq)))
        else:
            df = 1.0
        spectrum_power_db = _power_to_level_db(psd_w * max(df, 1e-30))

        if pr_spectrum.size and np.any(~np.isnan(pr_spectrum)):
            i_max = int(np.nanargmax(pr_spectrum))
            max_pr_db = float(pr_spectrum[i_max])
            max_pr_freq = float(f_axis[i_max])
        else:
            max_pr_db = float("nan")
            max_pr_freq = float("nan")

        status, overall_ok, max_exceed, no_valid = self._decide(tones, params)

        return ProminenceRatioSpectrumResult(
            frequency_hz=f_axis,
            band_power_db=band_power_db,
            pr_db=pr_spectrum,
            fft_magnitude_db=spectrum_power_db,
            fft_freq_hz=freq,
            main_tones=tones,
            max_pr_db=max_pr_db,
            max_pr_frequency_hz=max_pr_freq,
            decision_status=status,
            overall_ok=overall_ok,
            max_exceed_db=max_exceed,
            no_valid_main_tone=no_valid,
            warnings=warnings,
            metadata=metadata,
        )
