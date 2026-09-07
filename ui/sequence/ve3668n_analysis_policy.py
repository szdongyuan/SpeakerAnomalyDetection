"""File/recording-local VE admission. Never consult current hardware or stores."""
from dataclasses import dataclass

import numpy as np

from base.ve3668n_wav_metadata import resolve_ve_wav_channel_v2pa_factor, validate_ve_wav_metadata
from base.wav_calibration_metadata import WavCalibrationMetadataReadStatus


@dataclass(frozen=True)
class VEAnalysisDecision:
    columns: tuple[int, ...]
    factors: tuple[float, ...] = ()
    diagnostic: str = ""

    @property
    def allowed(self):
        return bool(self.factors) and not self.diagnostic


def verified_recorded_channels(source):
    """Expose physical mapping only after the complete source validates."""
    if source.status is not WavCalibrationMetadataReadStatus.VALID:
        return None
    try:
        normalized = validate_ve_wav_metadata(source.metadata)
    except ValueError:
        return None
    return tuple(item["physical_input_channel"] for item in sorted(
        normalized["recorded_channels"], key=lambda item: item["wav_channel_index"]))


def resolve_ve_analysis(source, columns, *, column_count):
    """Approve ALL actual input columns, or return one actionable diagnostic.

    ``source`` is Task5's diagnostic (also used for validated Task9 snapshots).
    Column order is the audio array order, never a current device selection.
    """
    columns = tuple(columns)
    if source.status is not WavCalibrationMetadataReadStatus.VALID:
        return VEAnalysisDecision(columns, diagnostic="VE 元数据无效，无法确认实测校准")
    if not columns or any(type(col) is not int or not 0 <= col < column_count for col in columns):
        return VEAnalysisDecision(columns, diagnostic="VE 分析通道不存在，无法确认实测校准")
    factors = []
    for column in columns:
        resolution = resolve_ve_wav_channel_v2pa_factor(source.metadata, column)
        if resolution.factor is None:
            return VEAnalysisDecision(columns, diagnostic=(
                f"WAV 通道 {column + 1} 缺少有效实测校准，需先完成输入校准："
                f"{resolution.diagnostic}"))
        factors.append(resolution.factor)
    if len(source.metadata["recorded_channels"]) != column_count:
        return VEAnalysisDecision(columns, diagnostic="VE 元数据与音频通道数不一致，校准无效")
    return VEAnalysisDecision(columns, tuple(factors))


class VEPressureSignalView:
    """Bounded, independent signal inputs for Spec and the joint PD/ED path.

    Not a DataDealStruct copy/constructor: that class is a singleton. Result
    maps intentionally remain shared, while signal arrays and source scalars
    belong to this analysis. Only one pressure vector is retained: the approved
    Spec column, or the mean of individually converted PD/ED input columns.
    Consumers use mono calculation index 0 and factor=1 for this Pa input.
    """
    def __init__(self, source, decision):
        if not decision.allowed:
            raise ValueError(decision.diagnostic or "VE pressure input was not approved")
        raw = source.store_wave_data_multi
        if raw is None:
            raw = np.asarray(source.store_wave_data).reshape(-1, 1)
        pressure = np.multiply(raw[:, decision.columns[0]], decision.factors[0], dtype=np.float64)
        for column, factor in zip(decision.columns[1:], decision.factors[1:]):
            pressure += np.multiply(raw[:, column], factor, dtype=np.float64)
        if len(decision.columns) > 1:
            pressure /= len(decision.columns)
        self.store_wave_data_multi = None
        self.store_wave_data = pressure
        self.sample_rate = source.sample_rate
        self.audio_lenth = len(pressure)
        self.stimulus_data = getattr(source, "stimulus_data", None)
        self.stimulus_info = getattr(source, "stimulus_info", None)
        self.analysis_result_dict = source.analysis_result_dict
        self.pd_peak_grid_points_map = source.pd_peak_grid_points_map
