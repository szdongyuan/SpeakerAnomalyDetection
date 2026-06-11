import ast
import types
from datetime import datetime
from pathlib import Path

from base.pdf_result_exporter import PdfExportResult


REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_WIDGET_PATH = REPO_ROOT / "ui" / "sequence" / "sequence_widget.py"


def _bind_sequence_methods(window, *method_names, namespace=None):
    source = SEQUENCE_WIDGET_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow")
    ns = {"datetime": datetime}
    if namespace:
        ns.update(namespace)
    for method_name in method_names:
        method_node = next(
            node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        module = ast.Module(body=[method_node], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(SEQUENCE_WIDGET_PATH), "exec"), ns)
        setattr(window, method_name, ns[method_name].__get__(window, type(window)))


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, message):
        self.infos.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))

    def error(self, message):
        self.errors.append(str(message))


class FakeLineEdit:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class FakeAnalysis:
    _sequence_analysis_key = "SPL1"
    result = {"score": 1}
    export_detail = {"summary": "done"}
    pdf_summary_exclude_fields = ("score",)

    def export_pdf_images(self, output_dir):
        return []

    def export_pdf_tables(self):
        return [{"title": "分析表格", "headers": ["项目"], "rows": [["值"]]}]


def _new_window():
    window = types.SimpleNamespace()
    window.default_logger = FakeLogger()
    return window


def test_select_pdf_export_config_uses_first_enabled_in_display_sequence():
    window = _new_window()
    _bind_sequence_methods(window, "_select_pdf_export_config")
    window.analysis_config = {
        "display_sequence": ["PDF1", "PDF2"],
        "PDF1": {"type": "PDF", "enabled": True, "save_dir": "D:/out", "save_items": ["SPL1"]},
        "PDF2": {"type": "PDF", "enabled": True, "save_dir": "D:/out2", "save_items": ["FR1"]},
    }

    assert window._select_pdf_export_config()[0] == "PDF1"


def test_select_pdf_export_config_skips_disabled_and_preserves_sequence_order():
    window = _new_window()
    _bind_sequence_methods(window, "_select_pdf_export_config")
    window.analysis_config = {
        "display_sequence": ["PDF1", "SPL1", "PDF2"],
        "PDF1": {"type": "PDF", "enabled": False, "save_dir": "D:/out", "save_items": ["SPL1"]},
        "PDF2": {"type": "PDF", "enabled": True, "save_dir": "D:/out2", "save_items": ["FR1"]},
    }

    assert window._select_pdf_export_config()[0] == "PDF2"


def test_capture_analysis_export_cache_includes_pdf_image_callbacks():
    window = _new_window()
    _bind_sequence_methods(window, "_capture_analysis_export_cache")
    window.recorded_signal_info = {"file_path": "D:/records/demo.wav", "barcode": "SN001"}
    window.recorded_path = None
    window.lineedit_type = FakeLineEdit("MODEL-A")
    window.analysis_config = {"SPL1": {"type": "SPL"}}
    window.analysis_window = [FakeAnalysis()]
    window.data_struct = types.SimpleNamespace(analysis_result_dict={"SPL1": (True, 0.0)})
    window._analysis_export_run_id = 7

    window._capture_analysis_export_cache()

    cache = window._excel_export_cache
    assert cache["record_id"] == "D:/records/demo.wav"
    assert cache["audio_path"] == "D:/records/demo.wav"
    assert cache["run_id"] == 7
    assert cache["analysis_items_data"]["SPL1"]["result"] == {"score": 1}
    assert cache["analysis_items_data"]["SPL1"]["summary"] == "done"
    assert cache["analysis_items_data"]["SPL1"]["tables"] == [
        {"title": "分析表格", "headers": ["项目"], "rows": [["值"]]}
    ]
    assert cache["analysis_items_data"]["SPL1"]["pdf_summary_exclude_fields"] == ["score"]
    assert "SPL1" in cache["image_exporters"]
    assert cache["image_exporters"]["SPL1"]("D:/tmp") == []


def test_maybe_export_pdf_results_exports_once_per_analysis_run():
    window = _new_window()
    exports = []

    def fake_export(pdf_cfg, **kwargs):
        exports.append((pdf_cfg, kwargs))
        return PdfExportResult(ok=True, message="ok", file_path="D:/out/demo_142308.pdf")

    _bind_sequence_methods(
        window,
        "_select_pdf_export_config",
        "_maybe_export_pdf_results",
        namespace={"export_analysis_to_pdf": fake_export, "MessageBox": DummyMessageBox},
    )
    window.analysis_config = {
        "display_sequence": ["PDF1"],
        "PDF1": {"type": "PDF", "enabled": True, "save_dir": "D:/out", "save_items": ["SPL1"]},
        "SPL1": {"type": "SPL"},
    }
    window.recorded_signal_info = {"file_path": "D:/records/demo.wav"}
    window.recorded_path = None
    window._analysis_export_run_id = 1
    window._pdf_exported_run_id = None
    window._excel_export_cache = {
        "record_id": "D:/records/demo.wav",
        "audio_path": "D:/records/demo.wav",
        "sn": "SN001",
        "product_model": "MODEL-A",
        "date_text": "2026/6/5 14:23:08",
        "analysis_items_data": {"SPL1": {"type": "SPL", "result": {"score": 1}}},
        "analysis_result_dict": {"SPL1": (True, 0.0)},
        "image_exporters": {"SPL1": lambda _output_dir: []},
        "run_id": 1,
    }
    window._maybe_export_pdf_results()
    window._maybe_export_pdf_results()
    window._analysis_export_run_id = 2
    window._excel_export_cache["run_id"] = 2
    window._maybe_export_pdf_results()

    assert len(exports) == 2
    assert exports[0][1]["audio_path"] == "D:/records/demo.wav"
    assert exports[0][1]["image_exporters"] == window._excel_export_cache["image_exporters"]
    assert window._pdf_exported_run_id == 2


def test_handle_post_analysis_exports_isolates_pdf_exception():
    window = _new_window()
    _bind_sequence_methods(window, "_handle_post_analysis_exports")
    calls = []
    window.default_logger = type(
        "Logger",
        (),
        {"warning": lambda self, msg: calls.append(("warning", msg))},
    )()
    window._capture_analysis_export_cache = lambda: calls.append("cache")
    window._maybe_write_mes_result = lambda: calls.append("mes")
    window._maybe_export_excel_results = lambda: calls.append("excel")

    def raise_pdf():
        calls.append("pdf")
        raise RuntimeError("pdf boom")

    window._maybe_export_pdf_results = raise_pdf

    window._handle_post_analysis_exports()

    assert calls[:4] == ["cache", "mes", "excel", "pdf"]
    assert any(
        item[0] == "warning" and "pdf" in item[1].lower()
        for item in calls
        if isinstance(item, tuple)
    )


class DummyMessageBox:
    AcceptRole = 0
    RejectRole = 1
    Warning = 2

    def __init__(self, *_args, **_kwargs):
        self._clicked = None

    def setIcon(self, *_args, **_kwargs):
        pass

    def setWindowTitle(self, *_args, **_kwargs):
        pass

    def setText(self, *_args, **_kwargs):
        pass

    def setInformativeText(self, *_args, **_kwargs):
        pass

    def addButton(self, label, role):
        button = (label, role)
        if self._clicked is None:
            self._clicked = button
        return button

    def setDefaultButton(self, *_args, **_kwargs):
        pass

    def exec_(self):
        pass

    def clickedButton(self):
        return self._clicked
