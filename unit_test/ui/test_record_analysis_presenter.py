"""Selection renders memory results only and releases old data."""

from copy import deepcopy
from datetime import datetime

from PyQt5.QtCore import QEvent

from base.motor_daily_wav_records import DailyWavRecord
from ui.sequence.record_analysis_dialog import RecordAnalysisDialog
from ui.sequence.record_analysis_presenter import RecordAnalysisPresenter
from ui.sequence.record_analysis_results import build_record_analysis_results
from unit_test.ui.test_record_analysis_results import real_results
from unit_test.ui.test_record_analysis_view_layout import text as label_text


def test_default_selection_switching_and_independent_current_judgement(qt_app, real_results):
    _, items, result = real_results
    dialog = RecordAnalysisDialog()
    dialog.show_record(DailyWavRecord(datetime(2026, 11, 16), 1, 'record.wav', '正转', 'NG'))
    presenter = RecordAnalysisPresenter(dialog)
    presenter.set_items(items)
    assert presenter.selected_key == 'SPL'
    assert dialog.status.text() == '待分析'
    presenter.show_status('分析中')
    dialog.navigation.setCurrentRow(1)
    assert dialog.status.text() == '分析中'
    current = deepcopy(result)
    for value in current['instance_results']:
        value.update(judgement='OK', judgement_status='completed', violations=[])
    presenter.show_result(build_record_analysis_results(items, current))
    old = dialog.content
    assert dialog.content.plots[0].record_panel_key == 'FFT'
    dialog.navigation.setCurrentRow(4)
    assert old.disposed
    assert dialog.channel_selector.count() == 2
    assert dialog.channel_selector.currentData() == 0
    assert label_text(dialog.current_result) == '本次判定：CH1 OK · CH2 OK'
    assert label_text(dialog.original_result) == '原记录结果：NG'
    dialog.channel_selector.setCurrentIndex(1)
    assert presenter.selected_channel == 1
    assert len(dialog.content.images) == 1
    presenter.dispose()
    assert dialog.content is None and not presenter.results
    dialog.navigation.setCurrentRow(0)
    assert dialog.content is None
    dialog.close()


def test_failed_heatmap_channel_remains_selectable_and_no_stale_graphics(qt_app, real_results):
    _, items, result = real_results
    broken = deepcopy(result)
    for value in broken['instance_results']:
        if value['analysis_type'] == 'Spec' and value['channel_index'] == 0:
            value.update(execution_status='分析失败', error={'message': 'missing calibration'})
        if value['analysis_type'] == 'FFT':
            value['display_payload'] = {'kind': 'wrong'}
    dialog = RecordAnalysisDialog()
    presenter = RecordAnalysisPresenter(dialog)
    presenter.set_items(items)
    presenter.show_result(build_record_analysis_results(items, broken))
    dialog.navigation.setCurrentRow(3)
    assert dialog.channel_selector.count() == 2
    assert not dialog.content.plots
    assert 'missing calibration' in dialog.content.diagnostic_text()
    dialog.channel_selector.setCurrentIndex(1)
    assert len(dialog.content.images) == 1
    old = dialog.content
    dialog.navigation.setCurrentRow(1)
    assert old.disposed and not dialog.content.plots
    assert '没有折线或频段绘图数据' in dialog.content.diagnostic_text()
    dialog.close()
    assert presenter.disposed and not presenter.results


def test_parent_destruction_releases_presenter_results(qt_app, real_results):
    from PyQt5.QtWidgets import QDialog
    parent = QDialog()
    dialog = RecordAnalysisDialog(parent)
    presenter = RecordAnalysisPresenter(dialog)
    _, items, result = real_results
    presenter.set_items(items)
    presenter.show_result(build_record_analysis_results(items, result))
    old = dialog.content
    curves = list(old.curves)
    parent.deleteLater()
    qt_app.sendPostedEvents(None, QEvent.DeferredDelete)
    assert presenter.disposed and not presenter.results and presenter.dialog is None
    assert old.disposed and all(curve.full_x.size == 0 for curve in curves)
