import ctypes
import sys

from base.db_upgrade import STATUS_ALREADY_UPGRADED, STATUS_SUCCESS, upgrade_legacy_single_database


def _show_message(title, message, is_error=False):
    if sys.platform.startswith("win"):
        style = 0x10 if is_error else 0x40
        ctypes.windll.user32.MessageBoxW(None, message, title, style)
    else:
        print(f"{title}: {message}")


def main():
    status, message = upgrade_legacy_single_database()
    is_error = status not in {STATUS_SUCCESS, STATUS_ALREADY_UPGRADED}
    title = "DB Upgrade Failed" if is_error else "DB Upgrade"
    _show_message(title, message, is_error=is_error)
    return 1 if is_error else 0


if __name__ == "__main__":
    sys.exit(main())
