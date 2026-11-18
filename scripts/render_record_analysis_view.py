"""Render synthetic, calibrated record-view evidence without accessing product data.

Run from any working directory:
    python scripts/render_record_analysis_view.py --output-dir PATH
All WAV/config/image/evidence files are owned by the supplied output directory.
The existing executor runs in a spawned process before the native Qt UI starts.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import sys

# Match other standalone scripts: make project imports independent of the CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare_fixture(output_dir):
    import numpy as np
    import soundfile as sf
    from base.record_analysis_reference import RecordAnalysisReference
    from base.record_analysis_request import build_record_analysis_request
    from base.analysis.input_adapter import build_analysis_batch
    from base.wav_calibration_metadata import append_wav_calibration_metadata
    from base.workbench_analysis_config import AnalysisTestQueue, WorkbenchAnalysisConfig
    from base.workbench_config_values import (
        WorkbenchChannel, WorkbenchCondition, WorkbenchContext,
        WorkbenchHardwareChannel, WorkbenchStation,
    )
    from base.workbench_recording_protocol import StationWavChannel

    rate = 44100
    time = np.arange(int(.4 * rate)) / rate
    signal = .02 * np.sin(2 * np.pi * 1000 * time) + .004 * np.sin(2 * np.pi * 2400 * time)
    wav = output_dir / ('合成验收记录_长文件名_' + 'record-analysis-' * 5 + '.wav')
    sf.write(wav, np.column_stack((signal, signal * 2)), rate, subtype='FLOAT')
    if not append_wav_calibration_metadata(wav, {'recorded_channels': [
        {'wav_channel_index': i, 'physical_input_channel': i, 'calibrated': True,
         'v2pa_factor': 1., 'standard_spl': 94.} for i in (0, 1)]}):
        raise RuntimeError('Synthetic WAV calibration metadata could not be written')
    channels = tuple(WorkbenchChannel(f'mic-{i}', name, 'sound_pressure',
        WorkbenchHardwareChannel('synthetic-offline-device', i))
        for i, name in enumerate(('近端麦克风', '远端麦克风')))
    values = {
        'SPL': {'type': 'SPL', 'analysis_channels': [0, 1], 'weighting': 'A',
                'limit_checked': True, 'limit_metric': 'overall_spl',
                'scalar_upper_enabled': True, 'scalar_upper_value': 60.,
                'scalar_lower_enabled': False},
        'FFT': {'type': 'FFT', 'analysis_channels': [0, 1], 'n_fft': 2048, 'x_axis_scale': 'linear'},
        'FBA': {'type': 'FBA', 'analysis_channels': [0, 1],
                'limit_checked': True, 'limit_metric': 'curve_y', 'limit_mode': 'manual',
                'manual_input_mode': 'constant', 'constant_upper_enabled': True,
                'constant_upper_value': 55., 'constant_lower_enabled': True,
                'constant_lower_value': -400.},
        'Spec': {'type': 'Spec', 'analysis_channels': [0, 1]},
        'LOUD': {'type': 'LOUD', 'analysis_channels': [0, 1],
                 'method': 'time_varying_iso532_1',
                 'advanced': {'curve_y_unit': 'phon', 'show_specific_loudness_heatmap': True},
                 'display': {'curves': ['loudness_time', 'specific_loudness_profile'],
                             'heatmaps': ['specific_loudness'],
                             'summary_metrics': ['steady_state_average_loudness', 'max_transient_loudness']}},
        '未支持项': {'type': 'AI', 'analysis_channels': [0, 1]},
    }
    for kind in ('SPL', 'FFT', 'FBA', 'LOUD'):
        values[kind].setdefault('display', {}).update(
            main_curve_color='#126798', upper_limit_color='#A26914', lower_limit_color='#723DA1')
    analysis = WorkbenchAnalysisConfig(WorkbenchContext('synthetic', '合成验收',
        str(output_dir / 'unused-results'), (WorkbenchStation('station', '合成工位', channels),),
        (WorkbenchCondition('condition', '正转', 'queue', ('station',), '', {'mode': 'none'}),)),
        {'queue': AnalysisTestQueue('wav', {'display_sequence': list(values), **values})})
    reference = RecordAnalysisReference('synthetic-recording', 'station', 'condition', str(wav),
        tuple(StationWavChannel(channel.channel_id, channel.channel_name, channel.signal_type,
                               'synthetic-device', i, i) for i, channel in enumerate(channels)))
    request = build_record_analysis_request(reference, analysis, 'synthetic-view')
    task, = build_analysis_batch(request.payload).tasks
    (output_dir / 'synthetic-config.json').write_text(
        json.dumps(request.payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return reference, request.items, task


def compute_fixture(task):
    from base.analysis.executor import execute_task
    return execute_task(task), os.getpid()


def data_ranges(result):
    import numpy as np

    def finite_range(values):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        return [float(finite.min()), float(finite.max())] if finite.size else None

    channels = []
    for channel in result.channels:
        panels = []
        for panel in channel.panels:
            entry = dict(key=panel.key, kind=panel.kind, x_unit=panel.x_unit, y_unit=panel.y_unit)
            if panel.heatmap is not None:
                entry.update(value_unit=panel.value_unit,
                    x=finite_range(panel.heatmap['x']), y=finite_range(panel.heatmap['y']),
                    z=finite_range(panel.heatmap['z']), shape=list(panel.heatmap['z'].shape))
            else:
                entry['series'] = [dict(name=series['name'], role=series['role'],
                    points=len(series['x']), x=finite_range(series['x']), y=finite_range(series['y']))
                    for series in panel.series]
            panels.append(entry)
        channels.append(dict(channel_index=channel.channel_index, wav_channel=channel.wav_channel,
            name=channel.name, status=channel.status, diagnostic=channel.diagnostic, panels=panels))
    return channels


def render_evidence(output_dir, reference, items, result, worker_pid):
    from dataclasses import replace
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    if Path('C:/Windows/Fonts').is_dir():
        os.environ['QT_QPA_FONTDIR'] = 'C:/Windows/Fonts'
    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtGui import QFont, QFontDatabase, QTextDocument
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from base.motor_daily_wav_records import DailyWavRecord
    from ui.sequence.record_analysis_dialog import RecordAnalysisDialog
    from ui.sequence.record_analysis_presenter import RecordAnalysisPresenter
    from ui.sequence.record_analysis_results import build_record_analysis_results

    app = QApplication.instance() or QApplication([])
    for name in ('msyh.ttc', 'simsun.ttc'):
        font = Path('C:/Windows/Fonts') / name
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    family = next((name for name in ('Microsoft YaHei', 'SimSun')
                   if name in QFontDatabase().families()), app.font().family())
    app.setFont(QFont(family, 10))
    mapped = build_record_analysis_results(items, result, channel_names={0: '近端麦克风', 1: '远端麦克风'})
    evidence = dict(source='synthetic calibrated stereo WAV; existing executor in spawned worker',
                    worker_pid=worker_pid, render_pid=os.getpid(), font=family,
                    analysis_display={item.config_key: item.parameters.get('display', {}) for item in items},
                    screenshots={})
    variants = (
        ('default-spl.png', 'SPL', (960, 680), 0),
        ('small-window.png', 'SPL', (720, 520), 0),
        ('multi-channel.png', 'FFT', (960, 680), 0),
        ('fba-single-channel.png', 'FBA', (1280, 760), 0),
        ('fba-multi-channel.png', 'FBA', (1280, 760), 0),
        ('spectrogram.png', 'Spec', (960, 680), 1),
        ('loudness-specific.png', 'LOUD', (960, 680), 0),
        ('failed-item.png', '未支持项', (960, 680), 0),
        ('single-channel-spl.png', 'SPL', (960, 680), 0),
        ('single-channel-ng.png', 'SPL', (960, 680), 1),
    )
    for filename, key, size, channel in variants:
        dialog = RecordAnalysisDialog()
        presenter = RecordAnalysisPresenter(dialog)
        try:
            dialog.show_record(DailyWavRecord(datetime(2026, 11, 16, 9, 30), 7,
                              Path(reference.wav_path).name, '合成正转工况', 'NG', reference))
            presenter.set_items(items)
            displayed = dict(mapped)
            if filename == 'fba-single-channel.png':
                displayed[key] = replace(mapped[key], channels=mapped[key].channels[:1])
            if filename.startswith('single-channel-'):
                displayed[key] = replace(mapped[key], channels=(mapped[key].channels[channel],))
            presenter.show_result(displayed)
            dialog.navigation.setCurrentRow(list(mapped).index(key))
            if dialog.channel_selector.count():
                dialog.channel_selector.setCurrentIndex(channel)
            # A virtual offscreen canvas can exceed its 800x600 default screen.
            # The product dialog itself still chooses its screen-bounded size.
            dialog.resize(*size)
            dialog.show()
            QTest.qWait(50)
            if key == 'LOUD':
                # Bring the full returned specific-loudness profile into view.
                profile = next(plot for plot in dialog.content.plots
                               if plot.record_panel_key == 'specific_loudness_profile')
                legend = dialog.content.content_layout.itemAt(
                    dialog.content.content_layout.indexOf(profile) - 1).widget()
                dialog.content_scroll.verticalScrollBar().setValue(legend.y())
                QTest.qWait(30)
            if not dialog.grab().save(str(output_dir / filename)):
                raise RuntimeError(f'Could not save {filename}')
            bounds = {}
            for name in ('navigation', 'content_scroll', 'file_name', 'channel_selector',
                         'original_result', 'current_result'):
                widget = getattr(dialog, name)
                point = widget.mapTo(dialog, widget.rect().topLeft())
                bounds[name] = [point.x(), point.y(), widget.width(), widget.height()]
            original, current = QTextDocument(), QTextDocument()
            original.setHtml(dialog.original_result.text())
            current.setHtml(dialog.current_result.text())
            evidence['screenshots'][filename] = dict(size=[dialog.width(), dialog.height()],
                bounds=bounds, selected_key=presenter.selected_key,
                original_result=original.toPlainText(), current_result=current.toPlainText(),
                selected_channel=presenter.selected_channel, status=dialog.status.text(),
                channels=data_ranges(displayed[key]),
                band_bars=[dict(x=bar.opts['x'].tolist(), height=bar.opts['height'].tolist(),
                               width=bar.opts['width'], channel=bar.toolTip())
                           for bar in dialog.content.bars],
                curve_styles=[dict(role=curve.record_role, color=curve.opts['pen'].color().name(),
                                   width=curve.opts['pen'].widthF(), style=int(curve.opts['pen'].style()))
                              for curve in dialog.content.curves],
                band_colors=[[brush.color().name() for brush in bar.opts['brushes']]
                             for bar in dialog.content.bars],
                ng_positions=[[text.pos().x(), text.pos().y()] for text in dialog.content.annotations],
                displayed_panels=[plot.record_panel_key for plot in dialog.content.plots],
                axes=[dict(key=plot.record_panel_key,
                           bottom=plot.getAxis('bottom').labelText,
                           left=plot.getAxis('left').labelText,
                           font=plot.getAxis('left').label.font().family(),
                           pixel_size=plot.getAxis('left').label.font().pixelSize())
                      for plot in dialog.content.plots],
                summary=[label.text() for label in dialog.content.summary_labels],
                scroll_maximum=dialog.content_scroll.verticalScrollBar().maximum(),
                plot_titles=[label.text() for label in dialog.content.title_labels],
                legends=[dict(component=type(legend).__name__, height=legend.height(), align_right=legend.align_right,
                              bounds=[legend.x(), legend.y(), legend.width(), legend.height()],
                              font=legend.font().family(), pixel_size=legend.font().pixelSize(),
                              overflow=legend.more.isVisible(), visible_count=legend._visible_count,
                              menu=[action.text() for action in legend.menu.actions()],
                              entries=[dict(name=name, color=pen.color().name(), width=pen.widthF(),
                                            style=int(pen.style())) for name, pen in legend.entries])
                         for legend in dialog.content.legends],
                plot_bounds={plot.record_panel_key: [plot.x(), plot.y(), plot.width(), plot.height()]
                             for plot in dialog.content.plots},
                scroll_value=dialog.content_scroll.verticalScrollBar().value())
        finally:
            dialog.close()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
    (output_dir / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    reference, items, task = prepare_fixture(directory)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
        result, worker_pid = executor.submit(compute_fixture, task).result(timeout=90)
    if result['execution_status'] not in ('分析完成', '结果不完整'):
        raise RuntimeError(result)
    for item in result['instance_results']:
        if item['analysis_type'] in ('SPL', 'FFT', 'FBA', 'Spec', 'LOUD'):
            if item['execution_status'] != '分析完成' or not item['display_payload']:
                raise RuntimeError(f'Supported synthetic analysis failed: {item}')
        if item['artifacts']:
            raise RuntimeError('Synthetic view unexpectedly persisted analysis artifacts')
    render_evidence(directory, reference, items, result, worker_pid)
    print(f'Rendered ten synthetic record-analysis views: {directory}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
