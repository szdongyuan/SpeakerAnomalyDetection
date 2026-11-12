"""Translate a completed task into one condition's curves, without file I/O."""

from dataclasses import dataclass

import numpy as np

from ui.sequence.station_plot_validation import validate_curve_series


@dataclass(frozen=True)
class ConditionPlotResult:
    series: tuple
    status: str = ''
    detail: str = ''
    heatmap: dict | None = None
    summaries: tuple = ()


def _exceedance_curve(instance, curve, channel, label, unit):
    """Display backend violation intervals; never rejudge sampled values."""
    if instance.get('judgement_status') != 'completed' or instance.get('judgement') != 'NG':
        return None
    mask = np.zeros(curve['x'].size, dtype=bool)
    for violation in instance.get('violations') or ():
        low, high = violation.get('x_start'), violation.get('x_end')
        if low is None and high is None:  # Overall SPL is not an instantaneous limit.
            continue
        if (not isinstance(low, (int, float)) or not isinstance(high, (int, float))
                or not np.isfinite([low, high]).all() or low > high
                or violation.get('x_unit') != unit):
            raise ValueError('超限区间或单位无效')
        mask |= (curve['x'] >= low) & (curve['x'] <= high)
    if mask.any():
        return dict(name=label, x=curve['x'], y=np.where(mask, curve['y'], np.nan),
                    wav_channel=channel, role='exceedance')
    return None


def _unit_matches(actual, expected):
    aliases = {f'dB({weight})': {f'dB({weight})', f'dB{weight}'} for weight in 'ABCDZ'}
    aliases['dB(Z)'].add('dB')
    return isinstance(actual, str) and actual in aliases.get(expected, {expected})


def _message(error, default):
    return str(error.get('message') or default) if isinstance(error, dict) else default


def build_condition_result(binding, segment, result):
    """Keep successful selected channels even when another instance fails.

    The controller has checked task identity. Units must match the shared axis;
    dBA/dB(A) and unweighted dB/dB(Z) are display spelling aliases only.
    """
    instances = result.get('instance_results', [])
    if not isinstance(instances, (list, tuple)):
        return ConditionPlotResult((), '结果格式错误', 'instance_results 必须是列表')
    series, issues, summaries = [], [], []
    heatmap = None
    for channel, name in segment.channels:
        matching = [item for item in instances if isinstance(item, dict)
                    and item.get('config_key') == binding.analysis_item
                    and item.get('analysis_type') == binding.analysis_type
                    and type(item.get('channel_index')) is int
                    and item['channel_index'] == channel - 1]
        label = f'CH{channel}'
        if name != label:
            label = f'{label} · {name}'
        if len(matching) != 1:
            issues.append(f'{label}：' + ('结果重复' if matching else _message(result.get('error'), '未返回该通道结果')))
            continue
        instance = matching[0]
        if instance.get('execution_status') != '分析完成':
            issues.append(f'{label}：{_message(instance.get("error"), "分析失败")}')
            continue
        payload = instance.get('display_payload')
        if binding.analysis_type == 'Spec':
            from ui.sequence.spectrogram_data import validate_spectrogram

            try:
                heatmap = validate_spectrogram(payload)
                heatmap['wav_channel'] = channel
            except (TypeError, ValueError) as error:
                issues.append(f'{label}：{error}')
            if instance.get('judgement_status') == 'failed':
                issues.append(f'{label}：{_message(instance.get("error"), "判定失败")}')
            continue
        if not isinstance(payload, dict) or payload.get('kind') != 'curve':
            issues.append(f'{label}：没有折线绘图数据')
            continue
        if payload.get('x_unit') != segment.x_unit or not _unit_matches(payload.get('y_unit'), segment.y_unit):
            issues.append(f'{label}：绘图单位与当前坐标轴不一致')
            continue
        raw = dict(name=label, x=payload.get('x'), y=payload.get('y'))
        try:
            curve, = validate_curve_series([raw], context=label, allow_gaps=True)
            if not curve['x'].size or not np.isfinite(curve['y']).any():
                raise ValueError('没有有效曲线数值')
        except (TypeError, ValueError) as error:
            issues.append(f'{label}：{error}')
            continue
        # Preserve original WAV-relative time/frequency and all valid points.
        series.append(dict(raw, wav_channel=channel, role='curve'))
        summaries.append(dict(channel=label, wav_channel=channel, judgement=instance.get('judgement'),
                              judgement_status=instance.get('judgement_status'),
                              metrics=instance.get('metrics') or {}))
        for side in ('upper', 'lower'):
            values = payload.get(side, [])
            limit = dict(name=label, x=payload['x'], y=values)
            try:
                if isinstance(values, (list, tuple, np.ndarray)) and len(values) == 0:
                    continue
                checked, = validate_curve_series([limit], context=f'{label} {side}', allow_gaps=True)
            except (TypeError, ValueError) as error:
                issues.append(f'{label} {side}：{error}')
                continue
            if np.isfinite(checked['y']).any():
                series.append(dict(limit, wav_channel=channel, role=side))
        try:
            highlight = _exceedance_curve(instance, curve, channel, label, segment.x_unit)
            if highlight is not None:
                series.append(highlight)
        except (TypeError, ValueError, AttributeError) as error:
            issues.append(f'{label}：{error}')
        if instance.get('judgement_status') == 'failed':
            issues.append(f'{label}：{_message(instance.get("error"), "判定失败")}')
    status = ('部分结果不可用' if series or heatmap is not None else '分析失败') if issues else ''
    return ConditionPlotResult(tuple(series), status, '\n'.join(issues), heatmap, tuple(summaries))
