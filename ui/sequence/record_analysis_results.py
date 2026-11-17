"""Pure, detached view data for a single record's complete backend results."""

from copy import deepcopy
from dataclasses import dataclass
import re

import numpy as np

from ui.sequence.spectrogram_data import validate_spectrogram
from ui.sequence.workbench_plot_display import ConditionSegmentDisplay
from ui.sequence.workbench_plot_results import build_condition_result
from ui.sequence.workbench_ui_projection import WorkbenchPlotBinding


@dataclass(frozen=True)
class RecordAnalysisPanel:
    """One axis pair, or a y-by-x matrix with a separate value/color unit."""

    key: str
    kind: str
    x_unit: str
    y_unit: str
    x_label: str = ''
    y_label: str = ''
    value_unit: str = ''
    value_label: str = ''
    series: tuple = ()
    heatmap: dict | None = None
    band_labels: tuple = ()
    metadata: dict | None = None


@dataclass(frozen=True)
class RecordAnalysisChannelResult:
    channel_index: int
    wav_channel: int
    name: str
    status: str
    diagnostic: str = ''
    execution_status: str | None = None
    judgement_status: str | None = None
    judgement: str | None = None
    metrics: dict | None = None
    violations: list | None = None
    panels: tuple[RecordAnalysisPanel, ...] = ()
    summary_cards: tuple = ()


@dataclass(frozen=True)
class RecordAnalysisItemResult:
    config_key: str
    analysis_type: str
    channels: tuple[RecordAnalysisChannelResult, ...]
    status: str
    diagnostic: str = ''


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} 必须是非空文本')
    return value


def _error_message(error, fallback):
    return str(error.get('message') or fallback) if isinstance(error, dict) else fallback


def _curve_panel(item, instance, channel_name, payload, *, key, kind='curve', metadata=None):
    x_unit = _text(payload.get('x_unit'), 'x_unit')
    y_unit = _text(payload.get('y_unit'), 'y_unit')
    channel = instance['channel_index'] + 1
    binding = WorkbenchPlotBinding('', '', item.config_key, item.analysis_type, (channel,))
    segment = ConditionSegmentDisplay('', '', channels=((channel, channel_name),),
                                      x_unit=x_unit, y_unit=y_unit)
    adapted = dict(instance, display_payload=dict(payload, kind='curve'))
    mapped = build_condition_result(binding, segment, {'instance_results': [adapted]})
    if not mapped.series:
        raise ValueError(mapped.detail or '没有有效曲线数值')
    series = tuple(dict(value, x=np.array(value['x'], dtype=float, copy=True),
                        y=np.array(value['y'], dtype=float, copy=True)) for value in mapped.series)
    labels = payload.get('band_labels', ()) if kind == 'bands' else ()
    if not isinstance(labels, (list, tuple)) or (kind == 'bands' and len(labels) != series[0]['x'].size):
        raise ValueError('band_labels 必须与频段中心数量一致')
    for label in labels:
        _text(label, 'band_labels')
    panel = RecordAnalysisPanel(
        key, kind, x_unit, y_unit, payload.get('x_label', ''), payload.get('y_label', ''),
        series=series, band_labels=tuple(labels), metadata=metadata,
    )
    return panel, mapped.detail


def _display_unit(value, axis):
    explicit = value.get(f'{axis}_unit')
    if explicit is not None:
        return _text(explicit, f'{axis}_unit')
    label = _text(value.get(f'{axis}_label'), f'{axis}_label')
    if ' / ' in label:
        return _text(label.split(' / ', 1)[1].strip('() '), f'{axis}_unit')
    match = re.search(r'\(([^()]+)\)$', label)
    if match:
        return match.group(1)
    if label in ('Bark', 'Hz', 's'):
        return label
    raise ValueError(f'{axis}_label 没有明确单位：{label}')


def _loudness_panels(item, instance, channel_name, payload):
    display = payload.get('loudness_display')
    if not isinstance(display, dict):
        raise ValueError('loudness_display 必须是字典')
    for field in ('curves', 'heatmaps', 'summary_cards'):
        if not isinstance(display.get(field), (list, tuple)):
            raise ValueError(f'loudness_display.{field} 必须是列表')
    panels, issues = [], []
    for value in display['curves']:
        if not isinstance(value, dict):
            raise TypeError('loudness_display.curves 条目必须是字典')
        key = _text(value.get('key'), 'curve.key')
        data = dict(value, x_unit=_display_unit(value, 'x'), y_unit=_display_unit(value, 'y'))
        # Only the time curve shares the backend main curve's limit domain.
        if key == 'loudness_time':
            data.update(upper=payload.get('upper', []), lower=payload.get('lower', []))
            current = instance
        else:
            current = dict(instance, judgement_status=None)
        metadata = {key: deepcopy(entry) for key, entry in value.items() if key not in ('x', 'y')}
        panel, detail = _curve_panel(item, current, channel_name, data, key=key, metadata=metadata)
        panels.append(panel)
        if detail:
            issues.append(detail)
    for value in display['heatmaps']:
        if not isinstance(value, dict):
            raise TypeError('loudness_display.heatmaps 条目必须是字典')
        key = _text(value.get('key'), 'heatmap.key')
        axes = [np.array(value.get(axis), dtype=float, copy=True) for axis in ('x', 'y')]
        for axis in axes:
            if (axis.ndim != 1 or not axis.size or not np.isfinite(axis).all()
                    or not np.all(np.diff(axis) > 0)):
                raise ValueError(f'{key} 热图坐标必须是严格递增的有限数组')
        z = np.array(value.get('z'), dtype=float, copy=True)
        if z.shape != (axes[1].size, axes[0].size) or not np.isfinite(z).all():
            raise ValueError(f'{key} 热图矩阵必须为 y × x 的有限数值')
        metadata = {field: deepcopy(entry) for field, entry in value.items() if field not in ('x', 'y', 'z')}
        panels.append(RecordAnalysisPanel(
            key, 'heatmap', _display_unit(value, 'x'), _display_unit(value, 'y'),
            value['x_label'], value['y_label'], _display_unit(value, 'z'), value['z_label'],
            heatmap=dict(x=axes[0], y=axes[1], z=z), metadata=metadata,
        ))
    cards = []
    for card in display['summary_cards']:
        if not isinstance(card, dict):
            raise TypeError('summary_cards 条目必须是字典')
        for field in ('key', 'label', 'unit'):
            _text(card.get(field), f'summary_cards.{field}')
        number = card.get('value')
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not np.isfinite(number):
            raise ValueError('summary_cards.value 必须是有限数值')
        cards.append(deepcopy(card))
    return tuple(panels), tuple(cards), '\n'.join(issues)


def _panels(item, instance, name):
    payload = instance.get('display_payload')
    if not isinstance(payload, dict):
        raise ValueError('display_payload 必须是字典')
    if item.analysis_type == 'LOUD':
        return _loudness_panels(item, instance, name, payload)
    if item.analysis_type == 'Spec':
        matrix = validate_spectrogram(payload)
        unit = _text(instance['metrics'].get('unit'), 'metrics.unit')
        panel = RecordAnalysisPanel(
            'Spec', 'heatmap', 's', 'Hz', value_unit=unit,
            heatmap=dict(x=matrix['time_s'], y=matrix['frequency_hz'], z=matrix['values_db']),
        )
        return (panel,), (), ''
    if payload.get('kind') not in ('curve', 'bands'):
        raise ValueError('没有折线或频段绘图数据')
    panel, detail = _curve_panel(item, instance, name, payload,
                                  key=item.analysis_type, kind=payload['kind'])
    return (panel,), (), detail


def build_record_analysis_results(items, result, *, channel_names=None):
    """Map exact (key, type, zero-based channel) matches, with no I/O or judging.

    ``channel_names`` optionally maps frozen zero-based WAV indices to names.
    Panels have independent axes; matrices use ``heatmap['z'][y, x]``. Returned
    containers/arrays own their data and may be released with the session.
    """
    names = channel_names if channel_names is not None else {}
    instances = result.get('instance_results')
    task_error = result.get('error')
    task_failure = (result.get('execution_status') == '分析失败' and task_error is not None)
    views = {}
    for item in items:
        channels = []
        for index in item.channel_indices:
            name = names.get(index, f'CH{index + 1}')
            fields = dict(channel_index=index, wav_channel=index + 1, name=name)
            location = f'{item.config_key} [{item.analysis_type}] CH{index + 1}'
            if task_failure:
                diagnostic = _error_message(task_error, '任务分析失败')
            elif not isinstance(instances, (list, tuple)):
                diagnostic = 'instance_results 必须是列表'
            else:
                matching = [value for value in instances if isinstance(value, dict)
                            and value.get('config_key') == item.config_key
                            and value.get('analysis_type') == item.analysis_type
                            and type(value.get('channel_index')) is int and value['channel_index'] == index]
                diagnostic = ('结果重复，通道结果不明确' if len(matching) > 1
                              else '未返回该通道结果' if not matching else '')
            if diagnostic:
                channels.append(RecordAnalysisChannelResult(
                    **fields, status='结果不可用', diagnostic=f'{location}：{diagnostic}'))
                continue
            instance = matching[0]
            fields.update(execution_status=instance.get('execution_status'),
                          judgement_status=instance.get('judgement_status'),
                          judgement=instance.get('judgement'), metrics=deepcopy(instance.get('metrics')),
                          violations=deepcopy(instance.get('violations')))
            if instance.get('execution_status') != '分析完成':
                channels.append(RecordAnalysisChannelResult(
                    **fields, status='分析失败', diagnostic=f'{location}：' +
                    _error_message(instance.get('error'), '分析失败')))
                continue
            # This is the external display-payload boundary. Shape/type errors
            # invalidate only this channel; programming exceptions still escape.
            try:
                if not isinstance(instance.get('metrics'), dict):
                    raise TypeError('metrics 必须是字典')
                panels, cards, detail = _panels(item, instance, name)
            except (TypeError, ValueError) as error:
                channels.append(RecordAnalysisChannelResult(
                    **fields, status='内容不可用', diagnostic=f'{location}：{error}'))
                continue
            if instance.get('error'):
                message = _error_message(instance['error'], '分析结果异常')
                if message not in detail:
                    detail = '\n'.join(filter(None, (detail, message)))
            if instance.get('judgement_status') == 'failed' and not detail:
                detail = '判定失败'
            channels.append(RecordAnalysisChannelResult(
                **fields, status='部分结果不可用' if detail else '分析完成',
                diagnostic=f'{location}：{detail}' if detail else '', panels=panels, summary_cards=cards))
        issues = [value.diagnostic for value in channels if value.diagnostic]
        status = ('部分结果不可用' if any(value.panels or value.summary_cards for value in channels)
                  else '结果不可用') if issues else '分析完成'
        views[item.config_key] = RecordAnalysisItemResult(
            item.config_key, item.analysis_type, tuple(channels), status, '\n'.join(issues))
    return views
