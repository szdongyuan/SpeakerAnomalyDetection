"""MoSQITo PR 参考后端（仅用于 ECMA 口径校核，不进入产品主路径）。

需求书《笔记本电脑风扇噪音核心测试要求书》4.2.4 要求：PR 结果需与 ECMA-74 PR
计算模块校核，偏差 ≤0.5 dB。MoSQITo 的 ``pr_ecma_*`` 实现 ECMA-418-1（即原
ECMA-74 Annex D，PR 公式一致）的 Prominence Ratio，可作为该校核基准。

约束：
- 仅适用于 ``ecma``（Zwicker 临界带 + Z 计权）口径。``customer_15pct``（15% 简化
  口径）是工程兼容口径，不得用 ECMA 参考结果对标，否则会得到误导性偏差。
- MoSQITo 未安装时优雅降级（``is_available() -> False``），调用方应跳过校核而非报错。
- 本模块只产出参考值，绝不覆盖项目主结果。
"""

from __future__ import annotations

import importlib.util
from typing import Optional

import numpy as np

_MOSQITO_AVAILABLE = importlib.util.find_spec("mosqito") is not None


def is_available() -> bool:
    """MoSQITo 是否可用（未安装时校核应跳过）。"""
    return _MOSQITO_AVAILABLE


def reference_pr_ecma(
    signal: np.ndarray,
    fs: int,
    *,
    mode: str = "ecma",
    prominence: bool = False,
) -> Optional[dict]:
    """用 MoSQITo 计算 ECMA-418-1 PR 作为校核基准。

    Parameters
    ----------
    signal : np.ndarray
        时域信号，单位 Pa（与项目内部一致，先乘 v2pa_factor 再传入）。
    fs : int
        采样率。
    mode : str
        必须为 ``"ecma"``；传入 ``customer_15pct`` 会抛 ``ValueError``，
        因为 15% 简化口径不得用 ECMA 参考对标。
    prominence : bool
        True 仅返回 prominent tone；False 返回全部检出 tone。

    Returns
    -------
    dict | None
        ``{"global_pr_db": float, "tones": [{"frequency_hz", "pr_db",
        "prominent"}...], "backend": "mosqito"}``；MoSQITo 不可用时返回 None。
    """
    if mode != "ecma":
        raise ValueError(
            "MoSQITo PR 参考仅适用于 ecma(Z) 口径；"
            "customer_15pct(15% 简化) 不得用 ECMA 参考对标。"
        )
    if not is_available():
        return None

    from mosqito.sq_metrics import pr_ecma_st

    sig = np.asarray(signal, dtype=float)
    t_pr, pr, promi, tones_freqs = pr_ecma_st(sig, fs, prominence=prominence)

    t_pr_arr = np.ravel(np.asarray(t_pr, dtype=float))
    global_pr = float(t_pr_arr[0]) if t_pr_arr.size else None

    tones = [
        {
            "frequency_hz": float(f),
            "pr_db": float(p),
            "prominent": bool(pm),
        }
        for f, p, pm in zip(np.atleast_1d(tones_freqs), np.atleast_1d(pr), np.atleast_1d(promi))
    ]
    return {
        "global_pr_db": global_pr,
        "tones": tones,
        "backend": "mosqito",
    }


def nearest_reference_tone(reference: dict, frequency_hz: float, tol_hz: float = 40.0) -> Optional[dict]:
    """在参考结果里取最接近 ``frequency_hz`` 的 tone（用于逐音对标）。"""
    if not reference or not reference.get("tones"):
        return None
    cands = [t for t in reference["tones"] if abs(t["frequency_hz"] - frequency_hz) <= tol_hz]
    if not cands:
        return None
    return min(cands, key=lambda t: abs(t["frequency_hz"] - frequency_hz))
