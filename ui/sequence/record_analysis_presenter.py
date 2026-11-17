"""Local item/channel selection and rendering of detached result models."""

from ui.sequence.record_analysis_renderers import RecordAnalysisRenderer


class RecordAnalysisPresenter:
    def __init__(self, dialog):
        self.dialog = dialog
        self.renderer = RecordAnalysisRenderer()
        self.items = {}
        self.results = {}
        self.selected_key = None
        self.selected_channel = None
        self.disposed = False
        self._status = ('待分析', '')
        dialog.item_selected.connect(self._select_item)
        dialog.channel_selected.connect(self._select_channel)
        dialog.view_closed.connect(self.dispose)
        dialog.destroyed.connect(self._view_destroyed)

    def set_items(self, items):
        if self.disposed:
            return
        items = tuple(items)
        self.items = {item.config_key: item for item in items}
        self.results = {}
        self.selected_key = None
        self.selected_channel = None
        self.dialog.set_analysis_items(items)
        if not items:
            self.show_status('当前工况未配置分析项')

    def show_result(self, results):
        """Accept Task 4's ordered config_key -> item-result mapping."""
        if not self.disposed:
            self.results = dict(results)
            self._render()

    def show_status(self, text, detail=''):
        if not self.disposed:
            self.results = {}
            self._status = (text, detail)
            self.dialog.show_status(text, detail)

    def _select_item(self, key):
        self.selected_key = key
        self.selected_channel = None
        self._render()

    def _select_channel(self, index):
        self.selected_channel = index
        self._render()

    def _render(self):
        if self.disposed:
            return
        item = self.items.get(self.selected_key)
        result = self.results.get(self.selected_key)
        if result is None:
            self.dialog.show_status(*self._status)
            return
        parameters = item.parameters
        heatmaps = (result.analysis_type == 'Spec'
                    or any(panel.kind == 'heatmap' for ch in result.channels for panel in ch.panels)
                    or (result.analysis_type == 'LOUD' and bool(parameters.get('display', {}).get('heatmaps'))))
        channels = result.channels if heatmaps else ()
        if channels and self.selected_channel not in {ch.channel_index for ch in channels}:
            self.selected_channel = channels[0].channel_index
        self.dialog.set_channels(channels, self.selected_channel)
        self.dialog.status.setText(result.status)
        # Channel diagnostics belong with their graphics/summary in the scroll area.
        self.dialog.status_detail.clear()
        self.dialog.status_detail.hide()
        self.dialog.set_content(self.renderer.render(
            result, parameters, channel_index=self.selected_channel if heatmaps else None))

    def _view_destroyed(self, *_):
        # Qt has already destroyed the child graphics; never call back into it.
        self.dialog = None
        self.results.clear()
        self.items.clear()
        self.disposed = True

    def dispose(self):
        if self.disposed:
            return
        self.disposed = True
        self.results.clear()
        self.items.clear()
        dialog, self.dialog = self.dialog, None
        dialog.item_selected.disconnect(self._select_item)
        dialog.channel_selected.disconnect(self._select_channel)
        dialog.view_closed.disconnect(self.dispose)
        dialog.destroyed.disconnect(self._view_destroyed)
        dialog.set_content(None)
