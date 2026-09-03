"""
Frequency Band Analysis — 频段能量分析模块

将音频信号的频谱按多种策略拆分为有限个频段，计算每个频段的声压级 (dB / dB(A))，
并与阈值进行比较。

支持的频段划分策略:
  - 1/N 倍频程 (ISO 266)
  - Bark 临界频带 (Zwicker)
  - 等宽频段
  - 自定义频段

支持的计权方式: A / C / Z (IEC 61672-1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.signal import get_window

from base.core_algorithm.response.frequency_weighting import get_weighting_fn


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Band:
    """频段定义"""
    f_low: float
    f_high: float
    f_center: float
    label: str


@dataclass
class SpectrumResult:
    """FFT 结果"""
    freqs: np.ndarray
    magnitude: np.ndarray
    psd: np.ndarray


@dataclass
class ThresholdResult:
    """单个频段的阈值比较结果"""
    band_label: str
    band_center: float
    value_db: float
    limit_db: float
    exceeded: bool
    margin_db: float


@dataclass
class BandAnalysisResult:
    """完整的频段分析结果"""
    bands: List[Band]
    band_levels_db: np.ndarray
    band_levels_weighted_db: np.ndarray
    overall_db: float
    overall_weighted_db: float
    weighting: str
    threshold_results: Optional[List[ThresholdResult]] = None
    exceeded_bands: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FFT / PSD 计算
# ---------------------------------------------------------------------------

def compute_spectrum(
    signal: np.ndarray,
    fs: int,
    window: str = "hann",
    nfft: Optional[int] = None,
) -> SpectrumResult:
    """
    对时域信号做 FFT，返回频率轴、幅值谱、功率谱密度。
    """
    N = len(signal)
    nfft = nfft or N

    if window == "rectangular":
        w = np.ones(N)
    else:
        w = get_window(window, N)

    # 去直流有助于避免 DC 泄漏污染低频段能量统计
    x = np.asarray(signal, dtype=np.float64)
    if x.size:
        x = x - float(np.mean(x))
    windowed = x * w

    spectrum = np.fft.rfft(windowed, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    magnitude = np.abs(spectrum) * 2.0 / np.sum(w)

    psd = (np.abs(spectrum) ** 2) / (fs * np.sum(w ** 2))
    psd[1:-1] *= 2

    return SpectrumResult(freqs=freqs, magnitude=magnitude, psd=psd)


# ---------------------------------------------------------------------------
# 频段划分策略
# ---------------------------------------------------------------------------

class BandStrategyType(str, Enum):
    OCTAVE = "octave"
    BARK = "bark"
    EQUAL_WIDTH = "equal_width"
    CUSTOM = "custom"


ISO_CENTERS_ONE_THIRD = [
    10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000,
]

BARK_BANDS = [
    (20, 100), (100, 200), (200, 300), (300, 400), (400, 510),
    (510, 630), (630, 770), (770, 920), (920, 1080), (1080, 1270),
    (1270, 1480), (1480, 1720), (1720, 2000), (2000, 2320), (2320, 2700),
    (2700, 3150), (3150, 3700), (3700, 4400), (4400, 5300), (5300, 6400),
    (6400, 7700), (7700, 9500), (9500, 12000), (12000, 15500),
]


def _format_freq(f: float) -> str:
    if f >= 1000:
        return f"{f / 1000:.4g}k"
    return f"{f:.4g}"


def parse_custom_bands(text):
    """Parse custom FBA band definitions from the configuration text."""
    edges = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = (
            [part.strip() for part in line.split(",") if part.strip()]
            if "," in line
            else [part for part in line.replace("\t", " ").split(" ") if part]
        )
        label = None
        try:
            if len(parts) == 1 and "-" in parts[0]:
                lower, upper = parts[0].split("-", 1)
                low, high = float(lower.strip()), float(upper.strip())
            elif len(parts) >= 2:
                low, high = float(parts[0]), float(parts[1])
                if len(parts) >= 3:
                    label = " ".join(parts[2:]).strip() or None
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"格式错误: {raw!r}") from exc
        if low <= 0 or high <= 0:
            raise ValueError(f"频率必须为正数: {raw!r}")
        if high <= low:
            raise ValueError(f"频段上限必须大于下限: {raw!r}")
        edges.append((low, high, label))
    edges.sort(key=lambda item: item[0])
    if not edges:
        raise ValueError("请至少输入一个频段")
    for index in range(1, len(edges)):
        if edges[index][0] < edges[index - 1][1]:
            raise ValueError("自定义频段不允许重叠，请检查相邻频段边界")
    return edges


def compute_octave_bands(f_min: float = 20, f_max: float = 20000, fraction: int = 3) -> List[Band]:
    """1/N 倍频程划分"""
    if fraction == 3:
        centers = [f for f in ISO_CENTERS_ONE_THIRD if f_min <= f <= f_max]
    else:
        ratio = 2 ** (1.0 / fraction)
        centers = []
        fc = 1000.0
        while fc >= f_min:
            centers.append(fc)
            fc /= ratio
        fc = 1000.0 * ratio
        while fc <= f_max:
            centers.append(fc)
            fc *= ratio
        centers = sorted(set(centers))

    factor = 2 ** (1.0 / (2 * fraction))
    bands = []
    for fc in centers:
        bands.append(Band(fc / factor, fc * factor, fc, _format_freq(fc)))
    return bands


def compute_bark_bands(f_min: float = 20, f_max: float = 15500) -> List[Band]:
    """Bark 临界频带划分 (Zwicker, 24 bands)"""
    bands = []
    for fl, fh in BARK_BANDS:
        if fh < f_min or fl > f_max:
            continue
        fl_clip = max(fl, f_min)
        fh_clip = min(fh, f_max)
        fc = np.sqrt(fl_clip * fh_clip)
        bands.append(Band(fl_clip, fh_clip, fc, _format_freq(fc)))
    return bands


def compute_equal_width_bands(f_min: float = 20, f_max: float = 20000, bandwidth: float = 100) -> List[Band]:
    """等宽频段划分"""
    bands = []
    f = f_min
    while f + bandwidth <= f_max:
        fl, fh = f, f + bandwidth
        fc = (fl + fh) / 2
        bands.append(Band(fl, fh, fc, _format_freq(fc)))
        f += bandwidth
    return bands


def compute_custom_bands(band_edges: List[Tuple[float, float, Optional[str]]]) -> List[Band]:
    """自定义频段划分"""
    bands = []
    for item in band_edges:
        fl, fh = item[0], item[1]
        label = item[2] if len(item) > 2 and item[2] else _format_freq(np.sqrt(fl * fh))
        bands.append(Band(fl, fh, np.sqrt(fl * fh), label))
    return bands


def get_bands(
    strategy: str,
    f_min: float = 20,
    f_max: float = 20000,
    fraction: int = 3,
    n_bands: int = 40,
    bandwidth: float = 100,
    custom_edges: Optional[List[Tuple[float, float, Optional[str]]]] = None,
) -> List[Band]:
    """根据策略名称获取频段列表"""
    strategy = strategy.lower()
    if strategy == "octave":
        return compute_octave_bands(f_min, f_max, fraction)
    elif strategy == "bark":
        return compute_bark_bands(f_min, f_max)
    elif strategy == "equal_width":
        return compute_equal_width_bands(f_min, f_max, bandwidth)
    elif strategy == "custom":
        return compute_custom_bands(custom_edges or [])
    else:
        raise ValueError(f"Unsupported band strategy: {strategy}")


# ---------------------------------------------------------------------------
# 阈值
# ---------------------------------------------------------------------------

class Threshold:
    """阈值定义：统一值或逐频段限值"""

    def __init__(self):
        self._limits: Dict[float, float] = {}
        self._uniform_limit: Optional[float] = None

    @classmethod
    def uniform(cls, limit_db: float) -> "Threshold":
        t = cls()
        t._uniform_limit = limit_db
        return t

    @classmethod
    def per_band(cls, limits: Dict[float, float]) -> "Threshold":
        t = cls()
        t._limits = limits
        return t

    @classmethod
    def from_csv_data(cls, freq_list, upper_list, lower_list=None) -> "Threshold":
        """从阈值曲线数据构建 (兼容 ThresholdConfigWidget 的数据格式)"""
        limits = {}
        for i, f in enumerate(freq_list):
            limits[float(f)] = float(upper_list[i])
        return cls.per_band(limits)

    def get_limit(self, f_center: float) -> float:
        if self._uniform_limit is not None:
            return self._uniform_limit
        if f_center in self._limits:
            return self._limits[f_center]
        freqs = sorted(self._limits.keys())
        if not freqs:
            return 0.0
        if f_center <= freqs[0]:
            return self._limits[freqs[0]]
        if f_center >= freqs[-1]:
            return self._limits[freqs[-1]]
        for i in range(len(freqs) - 1):
            if freqs[i] <= f_center <= freqs[i + 1]:
                ratio = (f_center - freqs[i]) / (freqs[i + 1] - freqs[i])
                return self._limits[freqs[i]] + ratio * (self._limits[freqs[i + 1]] - self._limits[freqs[i]])
        return 0.0

    def get_all_limits(self, centers: List[float]) -> np.ndarray:
        return np.array([self.get_limit(fc) for fc in centers])


# ---------------------------------------------------------------------------
# 主分析器
# ---------------------------------------------------------------------------

P_REF = 20e-6  # 参考声压 20μPa


class FrequencyBandAnalyzer:
    """
    频段能量分析器。

    用法:
        analyzer = FrequencyBandAnalyzer(strategy="octave", fraction=3, weighting="A")
        result = analyzer.analyze(signal, fs)
        result = analyzer.compare_threshold(result, threshold)
    """

    def __init__(
        self,
        strategy: str = "octave",
        weighting: str = "A",
        f_min: float = 20,
        f_max: float = 20000,
        window: str = "hann",
        fraction: int = 3,
        n_bands: int = 40,
        bandwidth: float = 100,
        custom_edges: Optional[List] = None,
    ):
        self.strategy = strategy
        self.weighting = weighting
        self.f_min = f_min
        self.f_max = f_max
        self.window = window
        self.fraction = fraction
        self.n_bands = n_bands
        self.bandwidth = bandwidth
        self.custom_edges = custom_edges

    def analyze(
        self,
        signal: np.ndarray,
        fs: int,
        p_ref: float = P_REF,
        v2pa_factor: float = 1.0,
    ) -> BandAnalysisResult:
        """
        执行频段能量分析。

        Args:
            signal: 一维时域音频信号
            fs: 采样率 (Hz)
            p_ref: 参考声压，默认 20μPa
            v2pa_factor: 电压→声压转换系数
        """
        signal = np.asarray(signal, dtype=np.float64) * float(v2pa_factor)

        spec = compute_spectrum(signal, fs, window=self.window)

        bands = get_bands(
            self.strategy,
            f_min=self.f_min,
            f_max=self.f_max,
            fraction=self.fraction,
            n_bands=self.n_bands,
            bandwidth=self.bandwidth,
            custom_edges=self.custom_edges,
        )

        if not bands:
            return BandAnalysisResult(
                bands=[], band_levels_db=np.array([]),
                band_levels_weighted_db=np.array([]),
                overall_db=-np.inf, overall_weighted_db=-np.inf,
                weighting=self.weighting,
            )

        if spec.freqs.size <= 1:
            return BandAnalysisResult(
                bands=bands,
                band_levels_db=np.full(len(bands), np.nan, dtype=np.float64),
                band_levels_weighted_db=np.full(len(bands), np.nan, dtype=np.float64),
                overall_db=np.nan,
                overall_weighted_db=np.nan,
                weighting=self.weighting,
            )

        df = float(spec.freqs[1] - spec.freqs[0])
        nyq = float(fs) / 2.0
        p_ref = float(p_ref)
        if p_ref <= 0.0 or not np.isfinite(p_ref):
            raise ValueError(f"p_ref must be finite positive, got {p_ref!r}")

        # 计权按“频点”作用到功率上再积分（更符合 IEC 计权定义，避免用中心频点近似带来的误差）
        weighting_fn = get_weighting_fn(self.weighting)
        corr_db_bins = weighting_fn(spec.freqs)
        weight_power_bins = 10.0 ** (corr_db_bins / 10.0)  # 功率域权重

        # overall: 直接对 [f_min, f_max] 的功率积分
        f_low_all = max(float(self.f_min), 0.0)
        f_high_all = min(float(self.f_max), nyq)
        idx_all = np.where((spec.freqs >= f_low_all) & (spec.freqs <= f_high_all))[0]
        if idx_all.size == 0:
            overall_power = 0.0
            overall_power_w = 0.0
        else:
            overall_power = float(np.sum(spec.psd[idx_all]) * df)
            overall_power_w = float(np.sum(spec.psd[idx_all] * weight_power_bins[idx_all]) * df)

        tiny = float(np.finfo(np.float64).tiny)
        overall_db = 10.0 * np.log10(max(overall_power, tiny) / (p_ref ** 2))
        overall_weighted_db = 10.0 * np.log10(max(overall_power_w, tiny) / (p_ref ** 2))

        band_levels_db = np.full(len(bands), np.nan, dtype=np.float64)
        band_levels_weighted_db = np.full(len(bands), np.nan, dtype=np.float64)

        # strategy-specific integration
        strategy = str(self.strategy).lower().strip()
        for i, band in enumerate(bands):
            fl = max(float(band.f_low), 0.0)
            fh = min(float(band.f_high), nyq)
            if fh <= fl:
                continue

            # 使用不重叠矩形带通积分
            idx = np.where((spec.freqs >= fl) & (spec.freqs < fh))[0]
            if idx.size == 0:
                fc = float(band.f_center)
                idx0 = int(np.argmin(np.abs(spec.freqs - fc)))
                if (spec.freqs[idx0] >= fl - df / 2.0) and (spec.freqs[idx0] < fh + df / 2.0):
                    idx = np.array([idx0], dtype=int)
                else:
                    continue
            power = float(np.sum(spec.psd[idx]) * df)
            power_w = float(np.sum(spec.psd[idx] * weight_power_bins[idx]) * df)

            band_levels_db[i] = 10.0 * np.log10(max(power, tiny) / (p_ref ** 2))
            band_levels_weighted_db[i] = 10.0 * np.log10(max(power_w, tiny) / (p_ref ** 2))

        return BandAnalysisResult(
            bands=bands,
            band_levels_db=band_levels_db,
            band_levels_weighted_db=band_levels_weighted_db,
            overall_db=overall_db,
            overall_weighted_db=overall_weighted_db,
            weighting=self.weighting,
        )

    @staticmethod
    def compare_threshold(
        result: BandAnalysisResult,
        threshold: Threshold,
        use_weighted: bool = True,
    ) -> BandAnalysisResult:
        """将分析结果与阈值进行比较。"""
        levels = result.band_levels_weighted_db if use_weighted else result.band_levels_db
        threshold_results = []
        exceeded_bands = []

        for i, band in enumerate(result.bands):
            limit = threshold.get_limit(band.f_center)
            value = float(levels[i])
            if np.isfinite(value):
                exceeded = bool(value > limit)
                margin = float(limit - value)
            else:
                exceeded = False
                margin = float("nan")
            threshold_results.append(ThresholdResult(
                band_label=band.label,
                band_center=band.f_center,
                value_db=value,
                limit_db=float(limit),
                exceeded=exceeded,
                margin_db=float(margin),
            ))
            if exceeded:
                exceeded_bands.append(i)

        result.threshold_results = threshold_results
        result.exceeded_bands = exceeded_bands
        return result
