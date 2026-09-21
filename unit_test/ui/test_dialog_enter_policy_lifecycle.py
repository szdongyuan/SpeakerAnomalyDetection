import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import traceback

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QDialog, QLineEdit, QVBoxLayout

from ui.dialog_enter_policy import install_dialog_enter_policy


def test_installed_policy_is_collected_without_callback_errors():
    script = textwrap.dedent("""
        import gc
        import json
        import sys
        import traceback
        import weakref
        from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit, QVBoxLayout
        from ui.dialog_enter_policy import install_dialog_enter_policy

        app = QApplication([])
        gc.disable()
        errors = []
        sys.excepthook = lambda *exc: errors.append(''.join(traceback.format_exception(*exc)))

        dialog = QDialog()
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLineEdit())
        install_dialog_enter_policy(dialog)
        refs = [weakref.ref(obj) for obj in (
            dialog, layout, dialog.findChild(QLineEdit),
            dialog._enter_policy, dialog._enter_policy._held_enter,
        )]
        del layout, dialog
        gc.collect()
        app.processEvents()
        gc.collect()
        print(json.dumps({'errors': errors, 'released': [ref() is None for ref in refs]}))
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["errors"] == [], outcome["errors"]
    assert all(outcome["released"]), outcome["released"]


@pytest.mark.parametrize("filter_name", ["policy", "held_enter"])
@pytest.mark.parametrize("key, modifiers", [
    (Qt.Key_Return, Qt.NoModifier),
    (Qt.Key_Enter, Qt.KeypadModifier),
])
@pytest.mark.parametrize("event_type", [QEvent.KeyPress, QEvent.KeyRelease])
def test_cleared_filter_state_ignores_delayed_autorepeat(
    ui_qapp, monkeypatch, filter_name, key, modifiers, event_type,
):
    received = []

    class Receiver(QLineEdit):
        def event(self, event):
            if event.type() == event_type:
                received.append((event.key(), event.isAutoRepeat()))
            return super().event(event)

    dialog = QDialog()
    child = QDialog(dialog)
    receiver = Receiver(child if filter_name == "held_enter" else dialog)
    policy = install_dialog_enter_policy(dialog)
    held_enter = policy._held_enter
    event_filter = policy if filter_name == "policy" else held_enter
    saved_state = event_filter.__dict__.copy()
    errors = []
    monkeypatch.setattr(
        sys, "excepthook",
        lambda *exc: errors.append("".join(traceback.format_exception(*exc))),
    )
    try:
        if filter_name == "held_enter":
            held_enter.arm()
        ui_qapp.postEvent(receiver, QKeyEvent(event_type, key, modifiers, "", True))
        event_filter.__dict__.clear()
        # Check the boundary's return value as well as actual queued Qt delivery.
        assert event_filter.eventFilter(
            receiver, QKeyEvent(event_type, key, modifiers, "", True),
        ) is False
        ui_qapp.processEvents()
        assert errors == [], errors
        assert received == [(key, True)]
    finally:
        event_filter.__dict__.update(saved_state)
        held_enter.disarm()
        sip.delete(dialog)
        ui_qapp.processEvents()
