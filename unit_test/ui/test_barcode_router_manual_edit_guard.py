import ast
import textwrap
import types
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[2] / "ui" / "sequence" / "barcode_router.py"
TARGET_METHODS = {
    "normalize_barcode",
    "should_auto_commit_barcode",
    "on_barcode_text_changed",
    "on_barcode_debounce_timeout",
}


def _build_router_namespace(monotonic_values=None):
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    router_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BarcodeRouter")

    method_sources = {}
    for node in router_class.body:
        if isinstance(node, ast.FunctionDef) and node.name in TARGET_METHODS:
            method_sources[node.name] = textwrap.dedent(ast.get_source_segment(source, node))

    monotonic_iter = iter(monotonic_values or [1000.0])

    class DummySignalBlocker:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    namespace = {
        "QSignalBlocker": DummySignalBlocker,
        "time": types.SimpleNamespace(monotonic=lambda: next(monotonic_iter)),
    }

    for method_name in TARGET_METHODS:
        exec(method_sources[method_name], namespace)

    return namespace


class FakeTimer:
    def __init__(self):
        self.active = False
        self.start_calls = 0

    def start(self):
        self.active = True
        self.start_calls += 1

    def stop(self):
        self.active = False

    def isActive(self):
        return self.active


class FakeLineEdit:
    def __init__(self, text=""):
        self._text = text
        self._enabled = True

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value

    def isEnabled(self):
        return self._enabled


class FakeCheckBox:
    def __init__(self, checked=True):
        self.checked = checked

    def isChecked(self):
        return self.checked


def _build_router(namespace, *, text="SN-1234567", manual_guard=False):
    class FakeRouter:
        pass

    ctx = types.SimpleNamespace()
    ctx.barcode_scanner_box = FakeCheckBox(checked=True)
    ctx.lineedit_s_or_n = FakeLineEdit(text)
    ctx._sn_textchange_manual_guard = manual_guard
    ctx._barcode_debounce_timer = FakeTimer()
    ctx._barcode_first_char_ts = None
    ctx._barcode_last_char_ts = None
    ctx._barcode_min_length_for_auto_commit = 7
    ctx._barcode_fast_input_max_seconds = 0.4
    ctx._barcode_capture_buffer = ""
    ctx._barcode_capture_first_ts = None
    ctx._barcode_capture_last_ts = None
    ctx._barcode_capture_target_lineedit = None
    ctx._barcode_capture_target_text = None
    ctx._barcode_capture_target_cursor_pos = None
    ctx.commits = []
    ctx._commit_barcode = lambda text, source="wedge": ctx.commits.append((text, source))

    router = FakeRouter()
    router.ctx = ctx
    for method_name in TARGET_METHODS:
        setattr(router, method_name, namespace[method_name].__get__(router, type(router)))
    return router, ctx


def test_manual_edit_guard_blocks_textchange_auto_commit():
    namespace = _build_router_namespace(monotonic_values=[10.0])
    router, ctx = _build_router(namespace, text="SN-1234567", manual_guard=True)

    router.on_barcode_text_changed(ctx.lineedit_s_or_n.text())
    router.on_barcode_debounce_timeout()

    assert ctx.commits == []
    assert ctx._barcode_first_char_ts is None
    assert ctx._barcode_last_char_ts is None
    assert ctx._barcode_debounce_timer.isActive() is False


def test_empty_text_clears_manual_edit_guard_for_next_scan():
    namespace = _build_router_namespace(monotonic_values=[10.0])
    router, ctx = _build_router(namespace, text="", manual_guard=True)

    router.on_barcode_text_changed("")

    assert ctx._sn_textchange_manual_guard is False
    assert ctx._barcode_debounce_timer.isActive() is False
