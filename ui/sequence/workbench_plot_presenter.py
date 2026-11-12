"""Route current recording/analysis events to configured plot regions on the UI thread."""

import logging

from ui.sequence.workbench_plot_results import build_condition_result
from ui.sequence.workbench_plot_demo import simulated_rms_series


class WorkbenchPlotPresenter:
    def __init__(self, grid):
        self._grid = grid
        self._targets = []
        self._demo_targets = []
        self._request_ids = {}
        self._started_request = None
        self._batches = {}

    def configure(self, projection, *, invalidate_requests):
        self._targets = []
        self._demo_targets = []
        if invalidate_requests:
            self._request_ids.clear()
            self._started_request = None
            self._batches.clear()
        if projection is None:
            return
        for row, cells in enumerate(projection.cells):
            for column, cell in enumerate(cells):
                if cell.condition_plot is None:
                    continue
                if cell.condition_plot.settings.get('data_source') == 'simulated':
                    for index, segment in enumerate(cell.condition_plot.conditions):
                        self._demo_targets.append((row, column, cell.condition_plot.station_id,
                                                   segment, index, cell.condition_plot.settings))
                    continue
                for binding, segment in zip(projection.bindings[row][column], cell.condition_plot.conditions):
                    if binding is not None:
                        self._targets.append((row, column, binding, segment))

    def matches_request(self, context):
        key = context['station_id'], context['condition_id']
        return self._request_ids.get(key) == context['request_id']

    def _state_targets(self):
        yield from ((row, column, binding.station_id, binding.condition_id)
                    for row, column, binding, _ in self._targets)
        yield from ((row, column, station_id, segment.condition_id)
                    for row, column, station_id, segment, _, _ in self._demo_targets)

    def _station_state(self, event, status):
        for row, column, station_id, condition_id in self._state_targets():
            if (station_id, condition_id) == (event['station_id'], event['condition_id']):
                self._grid.set_condition_state(row, column, condition_id, status)

    def recording_started(self, context, *, starts_round):
        if self._started_request == context.request_id:
            return
        self._started_request = context.request_id
        if starts_round:
            self._request_ids.clear()
            self._batches.clear()
            for row, column, _, condition_id in self._state_targets():
                self._grid.set_condition_state(row, column, condition_id, '待采集', clear=True)
        for station in context.analysis.stations_for(context.condition_id):
            self._request_ids[station.station_id, context.condition_id] = context.request_id
        self.recording_state(context, '采集中', clear=True)

    def recording_state(self, context, status, detail='', *, clear=False):
        for row, column, station_id, condition_id in self._state_targets():
            if (condition_id == context.condition_id
                    and self._request_ids.get((station_id, condition_id)) == context.request_id):
                self._grid.set_condition_state(row, column, condition_id, status, detail, clear=clear)

    def begin_batch(self, request_id, tasks):
        """Wait for terminal results of submitted stations, not a fixed four slots."""
        self._batches[request_id] = (tuple(task['task_id'] for task in tasks), {})

    def analysis_event(self, event):
        if not event.get('is_current', False):
            return
        kind = event['event_type']
        if kind not in {'task_started', 'task_result'}:
            return
        batch = self._batches.get(event.get('request_id'))
        if kind == 'task_result' and batch is not None:
            task_ids, results = batch
            results[event['task_id']] = event
            if not all(task_id in results for task_id in task_ids):
                self._station_state(event, '分析中')
                return
            del self._batches[event['request_id']]
            # One GUI update transaction. Analysis and subscriber delivery stay independent.
            updates_enabled = self._grid.updatesEnabled()
            self._grid.setUpdatesEnabled(False)
            try:
                for task_id in task_ids:
                    self._display_event(results[task_id])
            finally:
                self._grid.setUpdatesEnabled(updates_enabled)
            return
        self._display_event(event)

    def _display_event(self, event):
        kind = event['event_type']
        if kind == 'task_started':
            self._station_state(event, '分析中')
            return
        self._display_demo_result(event)
        for row, column, binding, segment in self._targets:
            if (binding.station_id, binding.condition_id) != (event['station_id'], event['condition_id']):
                continue
            result = event['payload']
            if not isinstance(result, dict) or any(
                    result.get(key) != event[key] for key in ('task_id', 'station_id', 'condition_id')):
                detail = '分析结果的任务、工位或工况标识与提交任务不一致'
                logging.getLogger('core').warning('Plot result %s: %s', event['task_id'], detail)
                self._grid.set_condition_state(row, column, binding.condition_id, '结果不匹配', detail, clear=True)
                continue
            update = build_condition_result(binding, segment, result)
            if binding.analysis_type == 'Spec':
                self._grid.set_condition_heatmap_data(row, column, binding.condition_id, update.heatmap,
                                                      status=update.status, detail=update.detail)
            else:
                self._grid.set_condition_curve_data(row, column, binding.condition_id, update.series,
                                                    status=update.status, detail=update.detail,
                                                    summaries=update.summaries)
            if update.detail:
                logging.getLogger('core').warning('Plot result %s/%s: %s', event['task_id'],
                                                  binding.analysis_item, update.detail)

    def _display_demo_result(self, event):
        # Samples follow a real started request, never configuration load or arbitrary events.
        if not self._demo_targets or event.get('request_id') is None or not self.matches_request(event):
            return
        result = event['payload']
        success = (isinstance(result, dict) and result.get('execution_status') == '分析完成'
                   and all(result.get(key) == event[key] for key in ('task_id', 'station_id', 'condition_id')))
        for row, column, station_id, segment, index, settings in self._demo_targets:
            if (station_id, segment.condition_id) != (event['station_id'], event['condition_id']):
                continue
            if success:
                self._grid.set_condition_curve_data(
                    row, column, segment.condition_id,
                    simulated_rms_series(segment, settings, column, index), status='模拟数据',
                    detail='仅用于展示，不参与测试判定或结果保存',
                )
            else:
                self._grid.set_condition_state(row, column, segment.condition_id, '未生成模拟数据',
                                                '当前工况未成功完成分析', clear=True)
