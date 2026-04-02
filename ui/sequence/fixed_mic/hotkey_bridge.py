import ctypes
from ctypes import wintypes

from PyQt5.QtCore import QObject, QEvent, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QShortcut

WINDOWS_MESSAGE_KEYDOWN = 0x0100
WINDOWS_MESSAGE_SYSKEYDOWN = 0x0104
WINDOWS_VK_CONTROL = 0x11
WINDOWS_VK_MENU = 0x12
SUPPORTED_QT_MODIFIER_MASK = int(Qt.ControlModifier | Qt.AltModifier | Qt.ShiftModifier | Qt.MetaModifier)

DEFAULT_FIXED_MIC_HOTKEY_CONFIG = [
    {"shortcut": "Ctrl+1", "channel_index": 1},
    {"shortcut": "Ctrl+2", "channel_index": 2},
    {"shortcut": "Ctrl+3", "channel_index": 3},
    {"shortcut": "Ctrl+4", "channel_index": 4},
]

SHORTCUT_MODIFIER_TOKEN_MAP = {
    "CTRL": int(Qt.ControlModifier),
    "CONTROL": int(Qt.ControlModifier),
    "ALT": int(Qt.AltModifier),
    "SHIFT": int(Qt.ShiftModifier),
    "META": int(Qt.MetaModifier),
    "WIN": int(Qt.MetaModifier),
    "WINDOWS": int(Qt.MetaModifier),
}


class FixedMicHotkeyEventFilter(QObject):
    def __init__(self, window):
        super(FixedMicHotkeyEventFilter, self).__init__(window)
        self.window = window

    def eventFilter(self, watched, event):
        if event.type() != QEvent.KeyPress:
            return False
        if getattr(event, "isAutoRepeat", lambda: False)():
            return False

        channel_index = resolve_fixed_mic_hotkey_channel_from_qt_event(event, self.window)
        if channel_index is None:
            return False

        if not getattr(self.window, "isVisible", lambda: False)():
            return False
        if not hasattr(self.window, "is_fixed_mic_mode") or not self.window.is_fixed_mic_mode():
            return False

        active_window = QApplication.activeWindow()
        if active_window is not None and self.window.window() is not active_window:
            return False

        dispatch_fixed_mic_hotkey(window=self.window, channel_index=channel_index)
        event.accept()
        return True


def get_default_fixed_mic_hotkey_config():
    return [dict(item) for item in DEFAULT_FIXED_MIC_HOTKEY_CONFIG]


def get_configured_fixed_mic_hotkey_items(window=None):
    default_items = get_default_fixed_mic_hotkey_config()
    if window is None:
        return default_items

    try:
        acq_detail = window.sequence_config[0]["seq1"]["acq"].get("detail", {})
    except Exception:
        return default_items

    configured_items = acq_detail.get("fixed_mic_hotkeys")
    if isinstance(configured_items, list) and configured_items:
        return configured_items
    return default_items


def resolve_fixed_mic_key_token_spec(key_token):
    normalized_key = str(key_token or "").strip().upper()
    if not normalized_key:
        return None

    if len(normalized_key) == 1 and "0" <= normalized_key <= "9":
        digit_value = ord(normalized_key) - ord("0")
        return {
            "qt_key": int(Qt.Key_0) + digit_value,
            "windows_vk": 0x30 + digit_value,
        }

    if len(normalized_key) == 1 and "A" <= normalized_key <= "Z":
        return {
            "qt_key": int(getattr(Qt, "Key_%s" % normalized_key)),
            "windows_vk": ord(normalized_key),
        }

    if normalized_key.startswith("F") and normalized_key[1:].isdigit():
        function_index = int(normalized_key[1:])
        if 1 <= function_index <= 12:
            return {
                "qt_key": int(Qt.Key_F1) + function_index - 1,
                "windows_vk": 0x70 + function_index - 1,
            }
    return None


def parse_fixed_mic_shortcut_spec(shortcut_text, channel_index):
    shortcut = str(shortcut_text or "").strip()
    if not shortcut:
        return None

    try:
        normalized_channel = int(channel_index)
    except (TypeError, ValueError):
        return None
    if normalized_channel <= 0:
        return None

    shortcut_tokens = [token.strip() for token in shortcut.split("+") if token.strip()]
    if len(shortcut_tokens) < 2:
        return None

    modifier_tokens = shortcut_tokens[:-1]
    key_token = shortcut_tokens[-1]
    qt_modifiers = 0
    windows_modifier_state = {"control": False, "alt": False}
    normalized_tokens = []
    seen_modifier_tokens = set()
    for token in modifier_tokens:
        normalized_token = str(token).strip().upper()
        qt_modifier = SHORTCUT_MODIFIER_TOKEN_MAP.get(normalized_token)
        if qt_modifier is None or normalized_token in seen_modifier_tokens:
            return None
        seen_modifier_tokens.add(normalized_token)
        qt_modifiers |= int(qt_modifier)
        normalized_tokens.append("Ctrl" if normalized_token in ("CTRL", "CONTROL") else normalized_token.title())
        if normalized_token in ("CTRL", "CONTROL"):
            windows_modifier_state["control"] = True
        elif normalized_token == "ALT":
            windows_modifier_state["alt"] = True

    key_spec = resolve_fixed_mic_key_token_spec(key_token)
    if key_spec is None:
        return None

    normalized_shortcut = "+".join(normalized_tokens + [str(key_token).strip().upper()])
    windows_messages = (
        (WINDOWS_MESSAGE_SYSKEYDOWN,) if windows_modifier_state["alt"] else (WINDOWS_MESSAGE_KEYDOWN,)
    )
    return {
        "shortcut": normalized_shortcut,
        "channel_index": normalized_channel,
        "qt_key": int(key_spec["qt_key"]),
        "qt_modifiers": int(qt_modifiers),
        "windows_vk": int(key_spec["windows_vk"]),
        "windows_messages": windows_messages,
        "windows_modifier_state": windows_modifier_state,
    }


def build_fixed_mic_hotkey_specs(window=None):
    hotkey_items = get_configured_fixed_mic_hotkey_items(window)
    parsed_specs = []
    for item in hotkey_items:
        if not isinstance(item, dict):
            continue
        parsed_spec = parse_fixed_mic_shortcut_spec(
            item.get("shortcut"),
            item.get("channel_index"),
        )
        if parsed_spec is None:
            continue
        parsed_specs.append(parsed_spec)

    if parsed_specs:
        return parsed_specs

    default_specs = []
    for item in get_default_fixed_mic_hotkey_config():
        parsed_spec = parse_fixed_mic_shortcut_spec(item.get("shortcut"), item.get("channel_index"))
        if parsed_spec is not None:
            default_specs.append(parsed_spec)
    return default_specs


def build_fixed_mic_hotkey_map(window=None):
    return {spec["shortcut"]: spec["channel_index"] for spec in build_fixed_mic_hotkey_specs(window)}


def resolve_fixed_mic_hotkey_channel_from_qt_key(key, window=None):
    for spec in build_fixed_mic_hotkey_specs(window):
        if spec["qt_key"] == key:
            return spec["channel_index"]
    return None


def resolve_fixed_mic_hotkey_channel_from_qt_event(event, window=None):
    normalized_modifiers = int(event.modifiers()) & SUPPORTED_QT_MODIFIER_MASK
    for spec in build_fixed_mic_hotkey_specs(window):
        if spec["qt_key"] != event.key():
            continue
        if normalized_modifiers != int(spec.get("qt_modifiers", 0)):
            continue
        return spec["channel_index"]
    return None


def resolve_fixed_mic_hotkey_channel_from_windows_vk(virtual_key, window=None):
    for spec in build_fixed_mic_hotkey_specs(window):
        if int(spec["windows_vk"]) == int(virtual_key):
            return spec["channel_index"]
    return None


def resolve_fixed_mic_hotkey_channel_from_windows_message(message_id, virtual_key, modifier_state=None, window=None):
    normalized_modifier_state = normalize_fixed_mic_windows_modifier_state(modifier_state)
    for spec in build_fixed_mic_hotkey_specs(window):
        if int(spec["windows_vk"]) != int(virtual_key):
            continue
        if int(message_id or 0) not in tuple(spec.get("windows_messages", ())):
            continue
        if normalized_modifier_state != normalize_fixed_mic_windows_modifier_state(
            spec.get("windows_modifier_state", {})
        ):
            continue
        return spec["channel_index"]
    return None


def normalize_fixed_mic_windows_modifier_state(modifier_state=None):
    state = modifier_state if isinstance(modifier_state, dict) else {}
    return {
        "control": bool(state.get("control", False)),
        "alt": bool(state.get("alt", False)),
    }


def get_fixed_mic_windows_key_pressed(virtual_key):
    user32 = getattr(getattr(ctypes, "windll", None), "user32", None)
    if user32 is None:
        return False
    return bool(user32.GetKeyState(int(virtual_key)) & 0x8000)


def get_fixed_mic_windows_modifier_state():
    return {
        "control": get_fixed_mic_windows_key_pressed(WINDOWS_VK_CONTROL),
        "alt": get_fixed_mic_windows_key_pressed(WINDOWS_VK_MENU),
    }


def parse_fixed_mic_windows_msg(message):
    try:
        return wintypes.MSG.from_address(int(message))
    except (TypeError, ValueError):
        return None


def handle_fixed_mic_native_hotkey(window, event_type, message):
    if event_type != "windows_generic_MSG":
        return False

    msg = parse_fixed_mic_windows_msg(message)
    if msg is None:
        return False

    return handle_fixed_mic_windows_hotkey(
        window,
        msg.message,
        msg.wParam,
        modifier_state=get_fixed_mic_windows_modifier_state(),
    )


def dispatch_fixed_mic_hotkey(window, channel_index):
    if channel_index is None:
        return False
    if hasattr(window, "_handle_fixed_mic_hotkey_trigger"):
        window._handle_fixed_mic_hotkey_trigger(channel_index)
        return True
    return False


def handle_fixed_mic_windows_hotkey(window, message_id, virtual_key, modifier_state=None):
    channel_index = resolve_fixed_mic_hotkey_channel_from_windows_message(
        message_id,
        virtual_key,
        modifier_state=modifier_state,
        window=window,
    )
    if channel_index is None:
        return False
    if not getattr(window, "isVisible", lambda: False)():
        return False
    if not hasattr(window, "is_fixed_mic_mode") or not window.is_fixed_mic_mode():
        return False
    return dispatch_fixed_mic_hotkey(window, channel_index)


def install_fixed_mic_hotkeys(window):
    uninstall_fixed_mic_hotkeys(window)
    shortcuts = []
    for hotkey_text, channel_index in build_fixed_mic_hotkey_map(window).items():
        shortcut = QShortcut(QKeySequence(hotkey_text), window)
        # Use application-wide shortcut context so the trigger still works
        # when focus is inside nested child widgets on the sequence page.
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.setEnabled(False)
        shortcut.activated.connect(
            lambda selected_channel=channel_index: handle_fixed_mic_hotkey_trigger(window, selected_channel)
        )
        shortcuts.append(shortcut)
    window.fixed_mic_hotkey_shortcuts = shortcuts

    app = QApplication.instance()
    if app is not None:
        event_filter = FixedMicHotkeyEventFilter(window)
        app.installEventFilter(event_filter)
        window.fixed_mic_hotkey_event_filter = event_filter
    else:
        window.fixed_mic_hotkey_event_filter = None


def set_fixed_mic_hotkeys_enabled(window, enabled):
    for shortcut in getattr(window, "fixed_mic_hotkey_shortcuts", []) or []:
        shortcut.setEnabled(bool(enabled))


def uninstall_fixed_mic_hotkeys(window):
    app = QApplication.instance()
    event_filter = getattr(window, "fixed_mic_hotkey_event_filter", None)
    if app is not None and event_filter is not None:
        try:
            app.removeEventFilter(event_filter)
        except Exception:
            pass
    window.fixed_mic_hotkey_event_filter = None
    for shortcut in getattr(window, "fixed_mic_hotkey_shortcuts", []) or []:
        try:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        except Exception:
            pass
    window.fixed_mic_hotkey_shortcuts = []


def handle_fixed_mic_hotkey_trigger(window, channel_index):
    dispatch_fixed_mic_hotkey(window, channel_index)
