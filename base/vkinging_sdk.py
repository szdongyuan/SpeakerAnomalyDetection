"""Instance-scoped VkDaq binding; importing/constructing performs no native I/O.

ABI source: VkDaqAssistant/x64/libvkdaq.h (cdecl), and the vendor
ContAccelSamps-IntClk-V example. No dependency on ignored reference scripts.
"""
import ctypes
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import re

from base.ve3668n_input import validate_sample_rate, validate_voltage_range
from consts.ve3668n_consts import VE_MAX_INPUT_CHANNELS, VE_RANGE_MAX, VE_RANGE_MIN


class VkDaqError(RuntimeError):
    """Native/loader/protocol failure; code is None when no SDK code exists."""

    def __init__(self, operation, code, detail):
        self.operation = operation
        self.code = code
        self.detail = detail
        super().__init__(f"{operation} (code={code}): {detail}")


def _text_argument(value, field):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{field} must be nonempty text without NUL")
    return value.encode("utf-8")


class VkDaqClient:
    """Lazy, single-owner client; use a context manager or explicitly close.

    DLL selection: explicit path (no fallback), VKDAQ_HOME/<arch> or HOME,
    then VkDaqAssistant under Program Files. Injected DLLs bypass this search.
    close() releases only this client's dependency-search handle, NOT tasks;
    the capture owner must stop/clear its tasks before closing the client.
    """

    def __init__(self, *, dll=None, dll_path=None):
        self._dll = dll
        self._dll_path = dll_path
        self._functions = {}
        self._dll_directory = None
        self._closed = False

    def _select_dll(self):
        bits = ctypes.sizeof(ctypes.c_void_p) * 8
        arch = "x64" if bits == 64 else "x86"
        if self._dll_path is not None:
            candidates = [Path(self._dll_path).absolute()]
        else:
            candidates = []
            sdk_home = os.environ.get("VKDAQ_HOME")
            if sdk_home:
                home_path = Path(sdk_home).absolute()
                candidates.extend((home_path / arch / "libvkdaq.dll",
                                   home_path / "libvkdaq.dll"))
            program_files = os.environ.get(
                "ProgramFiles" if bits == 64 else "ProgramFiles(x86)",
                "C:/Program Files" if bits == 64 else "C:/Program Files (x86)",
            )
            candidates.append(Path(program_files) / "VkDaqAssistant" / arch / "libvkdaq.dll")
        for path in candidates:
            try:
                exists = path.is_file()
            except OSError as exc:
                raise VkDaqError("load_sdk", getattr(exc, "winerror", None) or exc.errno,
                                 f"cannot inspect SDK path {path}: {exc}") from exc
            if exists:
                return path
        raise VkDaqError(
            "load_sdk", None,
            f"libvkdaq.dll not found for {bits}-bit Python; install VkDaqAssistant, "
            f"set VKDAQ_HOME or supply dll_path. Searched: {candidates}",
        )

    def _load(self):
        path = self._select_dll()
        try:
            self._dll_directory = os.add_dll_directory(str(path.parent))
            self._dll = ctypes.CDLL(str(path))
        except OSError as exc:
            code = getattr(exc, "winerror", None) or exc.errno
            if code == 193:
                reason = "DLL bitness/architecture mismatch"
            elif code in (126, 127):
                reason = "DLL dependencies missing or incompatible"
            else:
                reason = "DLL load failed"
            error = VkDaqError(
                "load_sdk", code,
                f"{reason}: {path}; Python is {ctypes.sizeof(ctypes.c_void_p) * 8}-bit. "
                f"Check the vendor installation and VKDAQ_HOME. {exc}",
            )
            self._close_after_failure(error)
            raise error from exc

    def close(self):
        """Release this client's search handle; idempotent, no native task I/O."""
        self._closed = True
        self._functions.clear()
        self._dll = None
        if self._dll_directory is not None:
            try:
                self._dll_directory.close()
            except OSError as exc:
                # Keep the handle for an explicit cleanup retry; native use is
                # already disabled, so no uncertain client can be reused.
                raise VkDaqError("close_sdk", getattr(exc, "winerror", None) or exc.errno,
                                 f"dependency directory cleanup failed: {exc}") from exc
            self._dll_directory = None

    def _close_after_failure(self, original):
        try:
            self.close()
        except VkDaqError as cleanup:
            original.add_note(str(cleanup))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            self._close_after_failure(exc_value)

    def _bind(self):
        if self._closed:
            raise VkDaqError("load_sdk", None, "client is closed")
        if self._functions:
            return
        if self._dll is None:
            self._load()
        char = ctypes.c_char_p
        output = ctypes.POINTER(ctypes.c_char)
        integer = ctypes.c_int32
        double = ctypes.c_double
        signatures = {
            "VkDaqGetLastErrorInfo": ([], char),
            "VkDaqGetDevices": ([output, output, integer], integer),
            "VkDaqGetChannels": ([char, output, integer], integer),
            "VkDaqGetDeviceAttribute": ([char, char, output, integer], integer),
            "VkDaqCreateTask": ([char], integer),
            "VkDaqCreateAIAccelChan": (
                [char, char, char, integer, double, double, integer, double, integer, char],
                integer,
            ),
            "VkDaqCfgSampClkTiming": (
                [char, integer, double, integer, integer, integer], integer,
            ),
            "VkDaqStartTask": ([char], integer),
            "VkDaqStopTask": ([char], integer),
            "VkDaqClearTask": ([char], integer),
            "VkDaqGetTaskData": (
                [char, ctypes.POINTER(double), integer, integer, double], integer,
            ),
        }
        functions = {}
        for name, (argtypes, restype) in signatures.items():
            try:
                function = getattr(self._dll, name)
            except AttributeError as exc:
                error = VkDaqError("bind_sdk", None, f"missing required symbol {name}")
                self._close_after_failure(error)
                raise error from exc
            function.argtypes = argtypes
            function.restype = restype
            functions[name] = function
        self._functions = functions

    def _native_error(self, operation, code):
        try:
            raw = self._functions["VkDaqGetLastErrorInfo"]()
        except (OSError, ctypes.ArgumentError) as exc:
            # Error-info is itself an external call. Preserve the first native
            # code and diagnose this secondary failure rather than mask it.
            return VkDaqError(operation, code, f"VkDaqGetLastErrorInfo failed: {exc}")
        if not raw:
            detail = "SDK supplied no error text"
        else:
            try:
                detail = raw.decode("utf-8")
            except UnicodeDecodeError:
                detail = f"SDK error text has invalid UTF-8: {raw!r}"
        return VkDaqError(operation, code, detail)

    def _invoke(self, operation, *args):
        self._bind()
        try:
            code = self._functions[operation](*args)
        except (OSError, ctypes.ArgumentError) as exc:
            raise VkDaqError(operation, None, f"native call failed: {exc}") from exc
        if type(code) is not int or not -(2**31) <= code < 2**31:
            raise VkDaqError(operation, None, f"invalid int32 return: {code!r}")
        return code

    def _call(self, operation, *args):
        code = self._invoke(operation, *args)
        if code < 0:
            raise self._native_error(operation, code)
        return code

    def _read_text(self, operation, *args, outputs=1):
        size = 4096
        while True:
            # Nonzero sentinel: success without writing a NUL is NOT empty text.
            buffers = [ctypes.create_string_buffer(b"\xff" * size, size)
                       for _ in range(outputs)]
            code = self._invoke(operation, *args, *buffers, size)
            if code == -10028 and size < 1024 * 1024:
                size *= 2
                continue
            if code < 0:
                raise self._native_error(operation, code)
            result = []
            for index, buffer in enumerate(buffers):
                raw = bytes(buffer)
                end = raw.find(b"\0")
                if end < 0:
                    raise VkDaqError(operation, code, f"output {index} missing NUL terminator")
                try:
                    result.append(raw[:end].decode("utf-8"))
                except UnicodeDecodeError as exc:
                    raise VkDaqError(operation, code, f"output {index} invalid UTF-8: {exc}") from exc
            return tuple(result)

    @staticmethod
    def _parse_list(text, operation):
        if text == "":
            return ()
        entries = tuple(entry.strip() for entry in text.split(","))
        if any(not entry or any(ord(char) < 32 or ord(char) == 127 for char in entry)
               for entry in entries):
            raise VkDaqError(operation, None, "malformed comma list: empty or invalid entry")
        return entries

    def get_devices(self):
        """Return ordered (address, SDK name) pairs; empty output means no devices."""
        operation = "VkDaqGetDevices"
        addresses, names = self._read_text(operation, outputs=2)
        addresses = self._parse_list(addresses, operation)
        names = self._parse_list(names, operation)
        if len(addresses) != len(names):
            raise VkDaqError(operation, None, "device address/name list counts differ")
        return tuple(zip(addresses, names))

    def get_channels(self, device):
        """Return SDK channel strings without filtering/reordering physical routes."""
        operation = "VkDaqGetChannels"
        text, = self._read_text(operation, _text_argument(device, "device"))
        return self._parse_list(text, operation)

    def get_device_attribute(self, device, attribute):
        """Return strictly decoded SDK text; interpretation belongs to the caller."""
        text, = self._read_text(
            "VkDaqGetDeviceAttribute", _text_argument(device, "device"),
            _text_argument(attribute, "attribute"),
        )
        return text

    def create_task(self, task):
        return self._call("VkDaqCreateTask", _text_argument(task, "task"))

    def start_task(self, task):
        return self._call("VkDaqStartTask", _text_argument(task, "task"))

    def stop_task(self, task):
        return self._call("VkDaqStopTask", _text_argument(task, "task"))

    def clear_task(self, task):
        return self._call("VkDaqClearTask", _text_argument(task, "task"))

    def create_iepe_voltage_channel(self, task, physical_channels, *,
                                    range_min=VE_RANGE_MIN, range_max=VE_RANGE_MAX):
        """Create IEPE channels returning V; sensor settings are not exposed."""
        minimum, maximum = validate_voltage_range(range_min, range_max)
        return self._call(
            "VkDaqCreateAIAccelChan", _text_argument(task, "task"),
            _text_argument(physical_channels, "physical_channels"), b"", 0,
            minimum, maximum, 4, 1000.0, 0, b"",
        )

    def configure_sample_clock(self, task, sample_rate):
        """Configure onboard continuous sampling; the vendor uses count=0.

        This does NOT verify hardware readback: the SDK may defer configuration
        until Start. The owner must call verify_actual_sample_rate afterwards.
        """
        rate = validate_sample_rate(sample_rate)
        return self._call("VkDaqCfgSampClkTiming", _text_argument(task, "task"),
                          0, float(rate), 1, 0, 0)

    def verify_actual_sample_rate(self, device, requested_rate):
        """Return the verified integer rate, or fail without rounding/resampling.

        Call once native configuration has taken effect (possibly after Start),
        ALWAYS before the capture owner publishes started. SamplingStatus is
        deliberately not queried: the known device does not support it.
        """
        rate = validate_sample_rate(requested_rate)
        text = self.get_device_attribute(device, "SamplingFrequency").strip()
        detail = f"SamplingFrequency {text!r} must be finite and exactly {rate} Hz"
        if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text):
            raise VkDaqError("verify_actual_sample_rate", None, detail)
        try:
            actual = Decimal(text)
        except InvalidOperation as exc:
            raise VkDaqError("verify_actual_sample_rate", None, detail) from exc
        if not actual.is_finite() or actual != rate:
            raise VkDaqError("verify_actual_sample_rate", None, detail)
        return rate

    def read_task_data(self, task, *, channel_count, samples_per_channel, timeout_seconds):
        """Return (owned double array, actual samples/channel), without conversion.

        The owner MUST supply the task's configured channel count. VE blocks
        are at most 2048 samples/channel and waits at most 0.2 seconds (spec 6.2).
        The array has capacity channel_count * samples_per_channel and survives
        subsequent reads/close. Zero and short reads are valid; unused capacity
        is NOT audio. No tail/layout interpretation or finite-value check occurs
        here. GROUP_BY_CHANNEL=1 is fixed; compact short-read channel stride
        still requires hardware verification before capture acceptance.
        """
        task_name = _text_argument(task, "task")
        if type(channel_count) is not int or not 1 <= channel_count <= VE_MAX_INPUT_CHANNELS:
            raise ValueError("channel_count must be an integer from 1 through 8")
        if type(samples_per_channel) is not int or not 1 <= samples_per_channel <= 2048:
            raise ValueError("samples_per_channel must be an integer from 1 through 2048")
        if type(timeout_seconds) not in (int, float) or not 0 <= timeout_seconds <= 0.2:
            raise ValueError("timeout_seconds must be finite and between 0 and 0.2")
        buffer = (ctypes.c_double * (channel_count * samples_per_channel))()
        # Unwritten samples must never masquerade as valid zero-voltage audio.
        buffer[:] = [float("nan")] * len(buffer)
        count = self._call("VkDaqGetTaskData", task_name, buffer,
                           samples_per_channel, 1, float(timeout_seconds))
        if count > samples_per_channel:
            raise VkDaqError("VkDaqGetTaskData", count,
                             f"sample count {count} exceeds requested {samples_per_channel}")
        return buffer, count
