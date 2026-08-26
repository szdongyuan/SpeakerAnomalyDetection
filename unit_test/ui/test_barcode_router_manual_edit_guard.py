"""Compatibility coverage for the thin BarcodeRouter trigger port."""

from types import SimpleNamespace

from ui.sequence.barcode_router import BarcodeRouter


def test_manual_edit_guard_is_owned_by_trigger_port():
    calls = []
    port = SimpleNamespace(
        handle_barcode_return_pressed=lambda: calls.append("return"),
        handle_barcode_text_changed=lambda text: calls.append(("text", text)),
        handle_barcode_debounce_timeout=lambda: calls.append("timeout"),
        handle_keypress=lambda obj, event: calls.append(("key", obj, event)),
    )
    router = BarcodeRouter(port)

    router.on_barcode_text_changed("SN-1234567")
    router.on_barcode_debounce_timeout()

    assert calls == [("text", "SN-1234567"), "timeout"]
    assert not hasattr(router, "ctx")


def test_empty_text_is_forwarded_to_trigger_port_for_guard_reset():
    changed = []
    port = SimpleNamespace(
        handle_barcode_return_pressed=lambda: None,
        handle_barcode_text_changed=changed.append,
        handle_barcode_debounce_timeout=lambda: None,
        handle_keypress=lambda _obj, _event: None,
    )
    router = BarcodeRouter(port)

    router.on_barcode_text_changed("")

    assert changed == [""]
