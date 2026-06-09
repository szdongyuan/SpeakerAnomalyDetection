"""ECMA-418-1 / 客户 15% 临界带纯函数层（Prominence Ratio 算法基础）。

本模块只放“无状态纯函数”，集中实现 ECMA-418-1:2022 的临界带几何、相邻带边界、
低频修正与模式绑定计权。`prominence_ratio_analyzer.py`
负责编排（FFT/PSD、主音检测、决策），不在本文件之外重复实现这些公式。

设计要点（与开发规划 4.1 一致）：
- 所有 ECMA 公式集中在此，便于逐位对标标准与单测（T15/T16/T17）。
- PR 为带功率比值，默认在线性/不计权(Z)功率域计算；A/C 仅作为显式高级覆盖项。
- 临界带宽 / 相邻带二次拟合系数来源：ECMA-418-1:2022 Clause 10 / 12。
参考公式编号映射见 docs/PR模块与PR频谱模块开发规划.md 4.1.4。
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from base.core_algorithm.response.frequency_band_analyzer import get_weighting_fn

# ---------------------------------------------------------------------------
# 常量（ECMA-418-1:2022 频率适用范围）
# ---------------------------------------------------------------------------
ECMA_FT_MIN_HZ = 89.1            # 三分之一倍频程 100 Hz 段下沿
ECMA_SCOPE_FT_MAX_HZ = 11220.0   # 三分之一倍频程 10 kHz 段上沿（适用范围）
ECMA_FORMULA_FIT_MAX_HZ = 11200.0  # 部分拟合公式/工具采用的上界（兼容用）
LOW_FREQ_CORRECTION_FT_HZ = 171.4  # ft <= 171.4 Hz 时启用下临界带截断修正
MIDDLE_BAND_ARITH_MAX_HZ = 500.0   # ft <= 500 Hz 用算术对称，否则用几何边界


class BandTriplet(NamedTuple):
    """单个候选主音的三连续临界带边界 (Hz)。

    lower=(f1L, f2L)，middle=(f1M, f2M)，upper=(f1U, f2U)。
    满足 f1L < f2L == f1M < f2M == f1U < f2U。
    """

    lower: tuple[float, float]
    middle: tuple[float, float]
    upper: tuple[float, float]
    valid: bool
    reason: Optional[str]


# ---------------------------------------------------------------------------
# 临界带宽与 middle band 边界
# ---------------------------------------------------------------------------
def critical_bandwidth_hz(ft: float) -> float:
    """Zwicker 临界带宽 Δfc(ft)（ECMA-418-1 Clause 10 / Formula 2）。

    Δfc(ft) = 25 + 75 * (1 + 1.4 * (ft / 1000)^2)^0.69

    校核值：Δfc(500)≈117.3 Hz，Δfc(1000)≈162.2 Hz，Δfc(5000)≈914 Hz。
    """
    ratio = ft / 1000.0
    return 25.0 + 75.0 * (1.0 + 1.4 * ratio * ratio) ** 0.69


def ecma_middle_band(ft: float, fit_max_hz: float = ECMA_FORMULA_FIT_MAX_HZ) -> tuple[float, float]:
    """目标 middle critical band 边界 (f1M, f2M)。

    ft <= 500 Hz：算术对称（Formulae 4/5）
        f1M = ft - Δfc/2, f2M = ft + Δfc/2
    500 < ft <= fit_max：几何边界（Formulae 7/8）
        f1M = (-Δfc + sqrt(Δfc^2 + 4*ft^2)) / 2, f2M = f1M + Δfc
    """
    dfc = critical_bandwidth_hz(ft)
    if ft <= MIDDLE_BAND_ARITH_MAX_HZ:
        half = dfc / 2.0
        return ft - half, ft + half
    f1m = (-dfc + np.sqrt(dfc * dfc + 4.0 * ft * ft)) / 2.0
    return f1m, f1m + dfc


def ecma_adjacent_bands(
    ft: float,
    f1m: float,
    f2m: float,
    fit_max_hz: float = ECMA_FORMULA_FIT_MAX_HZ,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """下/上相邻临界带边界，返回 ((f1L, f2L), (f1U, f2U))。

    连续性约束：f2L = f1M，f1U = f2M。

    下相邻带下沿 f1L（ECMA-418-1 Clause 12 二次拟合）：
        89.1 <= ft < 171.4:        f1L = 20.0
        171.4 <= ft <= 1600:       f1L = -149.5 + 1.001*ft - 6.90e-5*ft^2
        1600 < ft <= fit_max:      f1L = 6.8 + 0.806*ft - 8.20e-6*ft^2
    上相邻带上沿 f2U：
        89.1 <= ft <= 1600:        f2U = 149.5 + 1.035*ft + 7.70e-5*ft^2
        1600 < ft <= fit_max:      f2U = 3.3 + 1.215*ft + 2.16e-5*ft^2

    实现门禁：以上二次拟合系数须在编码后从干净 ECMA-418-1:2022 PDF 逐位复核。
    """
    f2l = f1m
    f1u = f2m

    if ft < LOW_FREQ_CORRECTION_FT_HZ:
        f1l = 20.0
    elif ft <= 1600.0:
        f1l = -149.5 + 1.001 * ft - 6.90e-5 * ft * ft
    else:
        f1l = 6.8 + 0.806 * ft - 8.20e-6 * ft * ft

    if ft <= 1600.0:
        f2u = 149.5 + 1.035 * ft + 7.70e-5 * ft * ft
    else:
        f2u = 3.3 + 1.215 * ft + 2.16e-5 * ft * ft

    return (f1l, f2l), (f1u, f2u)


def customer_15pct_bands(ft: float, ratio: float = 0.15) -> BandTriplet:
    """客户需求书 15% 简化临界带（非 ECMA 标准）。

    目标带宽 bw = ratio * ft；目标带 [ft-bw/2, ft+bw/2]；
    相邻带在目标带两侧连续布置，各占一个带宽。
    """
    bw = ratio * ft
    half = bw / 2.0
    f1m, f2m = ft - half, ft + half
    lower = (f1m - bw, f1m)
    upper = (f2m, f2m + bw)
    valid = lower[0] > 0.0
    reason = None if valid else "lower band below 0 Hz"
    return BandTriplet(lower=lower, middle=(f1m, f2m), upper=upper, valid=valid, reason=reason)


def get_band_triplet(
    ft: float,
    mode: str = "ecma",
    *,
    customer_band_ratio: float = 0.15,
    ft_min_hz: float = ECMA_FT_MIN_HZ,
    ft_max_hz: float = ECMA_SCOPE_FT_MAX_HZ,
    fit_max_hz: float = ECMA_FORMULA_FIT_MAX_HZ,
    nyquist_hz: Optional[float] = None,
) -> BandTriplet:
    """按模式返回某候选主音 ft 的三连续临界带边界。

    mode=="ecma"：ECMA-418-1 临界带 + 相邻带拟合；
    mode=="customer_15pct"：15% 简化带。

    若三带未完整落在 [0, nyquist] 内，valid=False（该点 PR 无效，不参与判定）。
    """
    if mode == "customer_15pct":
        triplet = customer_15pct_bands(ft, customer_band_ratio)
    else:
        if ft < ft_min_hz or ft > ft_max_hz:
            reason = f"tone frequency outside ECMA range {ft_min_hz:g}-{ft_max_hz:g} Hz"
            return BandTriplet(lower=(np.nan, np.nan), middle=(np.nan, np.nan), upper=(np.nan, np.nan),
                               valid=False, reason=reason)
        f1m, f2m = ecma_middle_band(ft, fit_max_hz)
        (f1l, f2l), (f1u, f2u) = ecma_adjacent_bands(ft, f1m, f2m, fit_max_hz)
        reason = None
        valid = True
        if not (f1l < f2l <= f1m + 1e-9 and f1m < f2m and f2m <= f1u + 1e-9 and f1u < f2u):
            valid = False
            reason = "non-monotonic band edges"
        triplet = BandTriplet(lower=(f1l, f2l), middle=(f1m, f2m), upper=(f1u, f2u), valid=valid, reason=reason)

    if not triplet.valid:
        return triplet

    if nyquist_hz is not None and triplet.upper[1] > nyquist_hz:
        return triplet._replace(valid=False, reason="upper band exceeds Nyquist")
    if triplet.lower[0] <= 0.0:
        return triplet._replace(valid=False, reason="lower band below 0 Hz")
    return triplet


def low_freq_xl_correction_factor(ft: float, f1l: float, f2l: float) -> float:
    """ft <= 171.4 Hz 时的下临界带截断修正因子（Formula 23）：100 / ΔfL。

    XL_corr = XL * factor；ft > 171.4 Hz 时返回 1.0（不修正）。
    """
    if ft > LOW_FREQ_CORRECTION_FT_HZ:
        return 1.0
    dfl = f2l - f1l
    if dfl <= 0.0:
        return 1.0
    return 100.0 / dfl


# ---------------------------------------------------------------------------
# ECMA prominent 判据（Clause 12.6 / Formulae 24/25）
# ---------------------------------------------------------------------------
def ecma_prominence_limit_db(ft: float, ft_max_hz: float = ECMA_SCOPE_FT_MAX_HZ) -> Optional[float]:
    """ECMA prominent tone 阈值。

    89.1 <= ft < 1000:   limit = 9 + 10*log10(1000/ft)   # Formula (24)
    1000 <= ft <= ft_max: limit = 9                        # Formula (25)
    超出适用范围返回 None（该点不做 ECMA prominent 判定，只能作 customer 展示）。
    """
    if ft < ECMA_FT_MIN_HZ or ft > ft_max_hz:
        return None
    if ft < 1000.0:
        return 9.0 + 10.0 * np.log10(1000.0 / ft)
    return 9.0


# ---------------------------------------------------------------------------
# 模式绑定计权
# ---------------------------------------------------------------------------
def resolve_weighting(mode: str, weighting: str) -> tuple[str, list[str]]:
    """解析 PR 计算使用的有效计权，返回 (effective_weighting, warnings)。

    PR = 目标带功率 / 相邻带平均功率，是一个比值；需求书 4.4.3 / 设备表把 PR 模块
    幅度标度明确写为 dB（而非 Octave/FFT 的 dB(A)）。因此 PR 默认在**线性/不计权**
    功率域计算（Z 计权在本工程即 flat 0 dB = 线性），与带宽模式无关。

    - "auto"（默认）→ "Z"（线性/不计权）。
    - 显式 A / C → 尊重该值（隐藏高级覆盖项，用于特殊对比需求），不再按模式强制。
    - 显式 Z / none → "Z"。
    """
    warnings: list[str] = []
    w = (weighting or "auto").strip().upper()

    if w in ("AUTO", "Z", "NONE"):
        return "Z", warnings
    if w in ("A", "C"):
        warnings.append(f"PR 默认线性(Z)功率；当前按高级配置使用 {w} 计权")
        return w, warnings
    warnings.append(f"未知计权 {weighting}，回退线性(Z)")
    return "Z", warnings


def weighting_correction_db(freq_hz: np.ndarray, weighting: str) -> np.ndarray:
    """返回各频点的计权修正值 (dB)。复用 frequency_band_analyzer 的计权曲线。"""
    return get_weighting_fn(weighting)(np.asarray(freq_hz, dtype=float))


def apply_weighting_to_psd(psd: np.ndarray, freq_hz: np.ndarray, weighting: str) -> np.ndarray:
    """在功率域施加计权：psd_weighted = psd * 10^(W_dB/10)。Z 计权为恒等。"""
    if (weighting or "Z").strip().upper() in ("Z", "NONE", "AUTO"):
        return np.asarray(psd, dtype=float)
    w_db = weighting_correction_db(freq_hz, weighting)
    return np.asarray(psd, dtype=float) * (10.0 ** (w_db / 10.0))
