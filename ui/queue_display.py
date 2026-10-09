"""Queue labels independent of persisted registry aliases."""

import os

from base.sequence_queue_references import queue_path_key
from consts.running_consts import DEFAULT_DIR


def is_default_queue_path(path):
    default_path = os.path.join(DEFAULT_DIR, "ui", "ui_config", "sequence_config.json")
    return bool(path) and queue_path_key(path, DEFAULT_DIR) == queue_path_key(default_path, DEFAULT_DIR)


def queue_display_name(name, path):
    return "默认配置" if is_default_queue_path(path) else name


def queue_display_options(catalog, names, current_queue):
    """Collapse default-file aliases while retaining an existing stored selection."""
    default_names = [
        name for name in names if is_default_queue_path(catalog[name].get("path"))
    ]
    selected_default = None
    if default_names:
        if current_queue in default_names:
            selected_default = current_queue
        elif "默认配置" in default_names:
            selected_default = "默认配置"
        else:
            selected_default = default_names[0]
    return sorted(
        (queue_display_name(name, catalog[name].get("path")), name)
        for name in names
        if name not in default_names or name == selected_default
    )
