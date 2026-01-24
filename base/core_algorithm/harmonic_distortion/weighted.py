import numpy as np
from typing import Literal, Tuple
from scipy.signal import bilinear, lfilter, filtfilt, freqz, firwin2, butter, sosfiltfilt


def _calibrate_1khz(b: np.ndarray, a: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    将滤波器在 1 kHz 处归一化为 0 dB（幅度 1）。

    参数:
    - b, a: IIR/FIR 滤波器系数
    - fs: 采样率 (Hz)

    返回:
    - (b_cal, a): 归一化后的分子系数与原分母系数
    """
    _, h1k = freqz(b, a, worN=[1000], fs=fs)
    gain = abs(h1k[0])
    return b / gain, a


def A_weighting_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """生成 A 计权滤波器的离散系数 (b, a)，在 1 kHz 处校准为 0 dB。"""
    "参考：https://en.wikipedia.org/wiki/A-weighting?utm_source=chatgpt.com"
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
    A1000 = 2.0
    NUM = [(2*np.pi*f4)**2 * (10**(A1000/20.0)), 0, 0, 0, 0]
    DEN = np.convolve([1, 4*np.pi*f4, (2*np.pi*f4)**2], [1, 4*np.pi*f1, (2*np.pi*f1)**2])
    DEN = np.convolve(DEN, [1, 2*np.pi*f2])
    DEN = np.convolve(DEN, [1, 2*np.pi*f3])
    b, a = bilinear(NUM, DEN, fs)
    return _calibrate_1khz(b, a, fs)


def B_weighting_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """生成 B 计权滤波器的离散系数 (b, a)，在 1 kHz 处校准为 0 dB。"""
    "参考：https://en.wikipedia.org/wiki/A-weighting?utm_source=chatgpt.com"
    f1, f2, f4 = 20.6, 158.5, 12194.0
    B1000 = 0.17
    NUM = [(2*np.pi*f4)**2 * (10**(B1000/20.0)), 0, 0, 0]
    DEN = np.convolve([1, 4*np.pi*f4, (2*np.pi*f4)**2], [1, 4*np.pi*f1, (2*np.pi*f1)**2])
    DEN = np.convolve(DEN, [1, 2*np.pi*f2])
    b, a = bilinear(NUM, DEN, fs)
    return _calibrate_1khz(b, a, fs)


def C_weighting_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """生成 C 计权滤波器的离散系数 (b, a)，在 1 kHz 处校准为 0 dB。"""
    "参考：https://en.wikipedia.org/wiki/A-weighting?utm_source=chatgpt.com"
    f1, f4 = 20.6, 12194.0
    C1000 = 0.06
    NUM = [(2*np.pi*f4)**2 * (10**(C1000/20.0)), 0, 0]
    DEN = np.convolve([1, 4*np.pi*f4, (2*np.pi*f4)**2], [1, 4*np.pi*f1, (2*np.pi*f1)**2])
    b, a = bilinear(NUM, DEN, fs)
    return _calibrate_1khz(b, a, fs)


def _d_weighting_wikipedia_mag(freqs_hz: np.ndarray) -> np.ndarray:
    """
    依据维基百科 D 计权幅频公式计算目标幅度曲线（未考虑相位），并在 1 kHz 处归一化。
    返回最小值限定为 1e-12 以避免数值问题。
    "参考：https://en.wikipedia.org/wiki/A-weighting?utm_source=chatgpt.com"
    """
    f = np.asarray(freqs_hz, dtype=float)
    f2 = f * f
    h = ((1037918.48 - f2)**2 + 1080768.16 * f2) / ((9837328 - f2)**2 + 11723776 * f2)
    denom = (f2 + 79919.29) * (f2 + 1345600.0)
    RD = (f / (6.8966888496476e-5)) * np.sqrt(h / denom)
    f0 = 1000.0
    f0_2 = f0 * f0
    h0 = ((1037918.48 - f0_2)**2 + 1080768.16 * f0_2) / ((9837328 - f0_2)**2 + 11723776 * f0_2)
    RD0 = (f0 / (6.8966888496476e-5)) * np.sqrt(h0 / ((f0_2 + 79919.29) * (f0_2 + 1345600.0)))
    mag = RD / RD0
    return np.maximum(mag, 1e-12)


def D_weighting_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 D 计权的等效 FIR 滤波器 (b, a)。
    说明: 使用 `firwin2` 拟合目标幅度曲线；零相位由 `filtfilt` 保证。
    """
    numtaps = 2049
    f = np.linspace(0.0, fs/2, numtaps)
    mag = _d_weighting_wikipedia_mag(f)
    b = firwin2(numtaps, f/(fs/2), mag, window='hamming')
    a = np.array([1.0])
    return _calibrate_1khz(b, a, fs)


def Z_weighting_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 Z 计权滤波器的离散系数 (b, a)。
    
    注意: Z 计权（零计权/平坦响应）滤波器目前尚未实现。
    此函数为占位符，返回单位滤波器（不改变信号）。
    
    待实现: 请根据 Z 计权的标准规范实现此函数。
    """
    # TODO: 实现 Z 计权滤波器
    # 目前返回单位滤波器（b=1, a=1），即不改变信号
    b = np.array([1.0])
    a = np.array([1.0])
    return b, a


def apply_weighting_filter(
    signal: np.ndarray,
    fs: float,
    weighting: Literal['A', 'B', 'C', 'D', 'Z', 'a', 'b', 'c', 'd', 'z'] = 'A',
    zero_phase: bool = True,
) -> np.ndarray:
    """
    对时域信号施加 A/B/C/D/Z 计权滤波。

    参数:
    - signal: 输入一维或二维数组；二维时按列为通道
    - fs: 采样率 (Hz)
    - weighting: 计权类型（不区分大小写），包括 A/B/C/D/Z
    - zero_phase: True 使用 `filtfilt` 零相位；False 使用 `lfilter`（可实时）

    返回:
    - 计权后的信号（与输入形状一致）
    """
    w = str(weighting).upper()
    if w == 'A':
        b, a = A_weighting_filter(fs)
    elif w == 'B':
        b, a = B_weighting_filter(fs)
    elif w == 'C':
        b, a = C_weighting_filter(fs)
    elif w == 'D':
        b, a = D_weighting_filter(fs)
    elif w == 'Z':
        b, a = Z_weighting_filter(fs)
    else:
        return signal

    if zero_phase:
        return filtfilt(b, a, signal)
    return lfilter(b, a, signal)



