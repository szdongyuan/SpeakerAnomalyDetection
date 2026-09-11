"""Read-only VE discovery policy and isolated asynchronous discovery."""
from collections import Counter
from dataclasses import dataclass
import importlib
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import threading
import time

from base import vkinging_sdk
from base.ve3668n_input import (
    create_input_config, normalize_machine_id, normalize_model,
    validate_device_snapshot, validate_physical_channels,
)
from consts.ve3668n_consts import VE_BACKEND


@dataclass(frozen=True)
class DiscoveryResult:
    """Verified snapshots only; failures/warnings never extend their schema."""

    devices: tuple = ()
    diagnostics: tuple = ()


def _text(value, field, *, empty=False):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    value = value.strip()
    if (not value and not empty) or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{field} must be valid text without control characters")
    return value


def _analog_channels(routes, name):
    if not isinstance(routes, (tuple, list)):
        raise ValueError("channels must be an ordered route list")
    channels = []
    for route in routes:
        if not isinstance(route, str) or not route.startswith(name + "/"):
            raise ValueError("channels route does not belong to enumerated device")
        terminal = route[len(name) + 1:]
        if re.fullmatch(r"(?:MIN|DIN|RAWAIN)[1-9][0-9]*", terminal):
            continue
        if not re.fullmatch(r"AIN[1-8]", terminal):
            raise ValueError("channels contains a malformed analog route")
        channels.append(int(terminal[3:]) - 1)
    return validate_physical_channels(channels)


def discover_devices(sdk):
    """Enumerate through a caller-owned SDK; no loading, tasks, or persistence.

    Attributes are queried ONLY by native name (the documented SDK contract).
    Simultaneously duplicate names cannot be disambiguated by address. IDs are
    checked across all verified identities before any record is made available.
    The new input_config is a discovery placeholder, not a restored profile.
    """
    diagnostics = []
    try:
        entries = sdk.get_devices()
    except vkinging_sdk.VkDaqError as exc:
        return DiscoveryResult(diagnostics=(str(exc),))
    if not isinstance(entries, (tuple, list)):
        return DiscoveryResult(diagnostics=("malformed device enumeration",))
    records = []
    for entry in entries:
        try:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("malformed device address/name pair")
            address, name = entry
            name = _text(name, "name")
            if "/" in name or "," in name:
                raise ValueError("malformed device name")
            records.append((address, name))
        except ValueError as exc:
            diagnostics.append(str(exc))
    aliases = Counter(name for _, name in records)
    candidates = []
    identities = []
    for address, name in records:
        if aliases[name] != 1:
            diagnostics.append(f"{name}: ambiguous native alias; not queried")
            continue
        try:
            machine_id = normalize_machine_id(_text(
                sdk.get_device_attribute(name, "MachineId"), "machine_id"))
            identities.append(machine_id)
            address = None if address is None else _text(address, "address", empty=True)
            model = normalize_model(sdk.get_device_attribute(name, "Model"))
            channels = _analog_channels(sdk.get_channels(name), name)
            device = validate_device_snapshot({
                "backend": VE_BACKEND, "model": model, "machine_id": machine_id,
                "name": name, "address": address, "physical_channels": channels,
                "max_input_channels": max(channels) + 1, "available": True,
                "input_config": create_input_config(),
            })
        except (vkinging_sdk.VkDaqError, ValueError) as exc:
            diagnostics.append(f"{name}: {exc}")
            continue
        try:
            _text(sdk.get_device_attribute(name, "DeviceStatus"), "DeviceStatus", empty=True)
        except (vkinging_sdk.VkDaqError, ValueError) as exc:
            diagnostics.append(f"{name}: optional DeviceStatus: {exc}")
        candidates.append(device)
    counts = Counter(identities)
    for machine_id, count in counts.items():
        if count != 1:
            diagnostics.append(f"machine_id {machine_id}: ambiguous stable identity")
    return DiscoveryResult(tuple(device for device in candidates
                                 if counts[device["machine_id"]] == 1), tuple(diagnostics))


def resolve_device(sdk, machine_id, channels):
    """Return a fresh route snapshot with channels in REQUEST order.

    Re-enumerate on every call. Capture must use its separately frozen request
    profile, never this snapshot's new-device default input_config.
    """
    machine_id = normalize_machine_id(_text(machine_id, "machine_id"))
    channels = validate_physical_channels(channels)
    result = discover_devices(sdk)
    matches = [device for device in result.devices if device["machine_id"] == machine_id]
    if len(matches) != 1:
        raise ValueError(f"machine_id {machine_id} unavailable: " + "; ".join(result.diagnostics))
    device = matches[0]
    if not set(channels).issubset(device["physical_channels"]):
        raise ValueError(f"machine_id {machine_id}: selected channels unavailable")
    return validate_device_snapshot({**device, "physical_channels": channels})


@dataclass(frozen=True)
class DiscoveryEvent:
    generation: int
    status: str
    result: DiscoveryResult
    child_pid: int | None
    message_bytes: int
    handles_released: bool


def _encode_result(result, limit):
    safe = _wire_result(result.devices, result.diagnostics, limit)
    payload = bytearray()
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    for chunk in encoder.iterencode({"devices": safe.devices, "diagnostics": safe.diagnostics}):
        if len(payload) + len(chunk) > limit:
            raise ValueError("discovery result exceeds byte limit")
        payload.extend(chunk.encode("ascii"))
    return bytes(payload)


def _wire_result(devices, diagnostics, limit):
    if not isinstance(devices, (tuple, list)) or not isinstance(diagnostics, (tuple, list)):
        raise ValueError("discovery protocol requires device and diagnostic lists")
    verified = []
    for item in devices:
        device = validate_device_snapshot(item)
        for field in ("name", "machine_id", "address"):
            if device[field] is not None:
                if len(device[field]) > limit:
                    raise ValueError("discovery result text exceeds byte limit")
                _text(device[field], field, empty=field == "address")
        if not device["available"] or "/" in device["name"] or "," in device["name"]:
            raise ValueError("discovery protocol contains an unverified device")
        verified.append(device)
    for field in ("name", "machine_id"):
        if len({item[field] for item in verified}) != len(verified):
            raise ValueError(f"discovery protocol has ambiguous {field}")
    if any(type(item) is not str for item in diagnostics):
        raise ValueError("discovery protocol diagnostics must be strings")
    if any(len(item) > limit for item in diagnostics):
        raise ValueError("discovery diagnostics exceed byte limit")
    return DiscoveryResult(tuple(verified), tuple(diagnostics))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("discovery protocol has duplicate JSON fields")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("discovery protocol contains non-finite numbers")


def _decode_result(payload, limit=65536):
    if type(payload) is not bytes or len(payload) > limit:
        raise ValueError("discovery protocol exceeds byte limit")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("discovery protocol nesting is excessive") from exc
    if type(value) is not dict or set(value) != {"devices", "diagnostics"}:
        raise ValueError("discovery protocol envelope fields are invalid")
    return _wire_result(value["devices"], value["diagnostics"], limit)


def _failure(stage, exc):
    # Bound diagnostics from native/import/protocol boundaries before encoding.
    return DiscoveryResult(diagnostics=(f"{stage}: {type(exc).__name__[:32]}: {str(exc)[:100]}",))


def _receive_one(connection, mailbox, limit):
    """A partial/broken native pipe message must not hold the supervisor hostage."""
    try:
        mailbox.put_nowait((connection.recv_bytes(limit), None))
    except (OSError, EOFError, ValueError) as exc:
        mailbox.put_nowait((None, exc))


def _discovery_worker(connection, factory_name, options_json, limit):
    """Only the spawn helper imports/constructs the SDK and makes native calls."""
    parent = multiprocessing.parent_process()
    finished = threading.Event()

    def parent_watch():
        while not finished.wait(.05):
            if parent is not None and not parent.is_alive():
                # Read-only discovery owns no acquisition tasks. A native probe,
                # constructor, or close can be stuck: do not call SDK cleanup on
                # this watcher thread. OS exit releases the helper's handles.
                os._exit(1)

    watcher = threading.Thread(target=parent_watch, name="ve-discovery-parent-watch", daemon=True)
    watcher.start()
    sdk = None
    result = DiscoveryResult()
    try:
        try:
            module, _, name = factory_name.rpartition(".")
            factory = getattr(importlib.import_module(module), name)
            sdk = factory(**json.loads(options_json))
            result = discover_devices(sdk)
        except Exception as exc:
            # Process contract boundary: arbitrary injected factory/import and
            # native failures must produce one diagnostic, not a fallback.
            result = _failure("discovery helper", exc)
        finally:
            if sdk is not None:
                try:
                    sdk.close()
                except Exception as exc:
                    # External SDK close boundary: preserve the probe's first
                    # failure, add cleanup context, and discard recording items.
                    # This helper is always retired, never reused after failure.
                    result = DiscoveryResult(diagnostics=(
                        *result.diagnostics, *_failure("SDK close", exc).diagnostics))
        try:
            payload = _encode_result(result, limit)
        except ValueError as exc:
            payload = _encode_result(_failure("discovery result", exc), limit)
        connection.send_bytes(payload)
    except (OSError, EOFError):
        # Parent cancellation/oversize rejection can close the pipe mid-send.
        # No more result is possible; logging diagnoses it and finally closes.
        logging.getLogger(__name__).warning("Discovery result pipe closed", exc_info=True)
    finally:
        connection.close()
        finished.set()
        watcher.join(.1)


@dataclass
class _ProbeResources:
    """One launch and all its handles, retained until every owner has stopped.

    Only launch() touches the Process while launcher is alive. The supervisor
    takes it back after joining that thread, including when start() raised.
    """

    receive: object = None
    send: object = None
    child: object = None
    launcher: threading.Thread | None = None
    receiver: threading.Thread | None = None
    launch_failure: DiscoveryResult | None = None

    def launch(self):
        try:
            self.child.start()
        except Exception as exc:
            # Process.start is an external boundary: OS setup and spawn's
            # serialization/import hooks can raise different exception types.
            # Preserve a bounded diagnostic, never accept the child's result,
            # and retain even a partially started child for supervisor cleanup.
            self.launch_failure = _failure("discovery launch", exc)

    @property
    def released(self):
        return all(handle is None for handle in (
            self.receive, self.send, self.child, self.launcher, self.receiver))


class DiscoveryService:
    """Instance-owned spawn supervisor, independent of Qt.

    start/refresh/cancel/close never join or perform SDK I/O. Poll the one-slot
    result mailbox, or provide a nonblocking on_result(event) that marshals to
    the GUI thread. The callback runs on the supervisor after bounded retirement;
    handles_released is false while launch or OS cleanup remains outstanding.
    At most one launcher/helper is owned; it must be reaped before another launch
    or an idle/closed claim. Already executing callbacks cannot be interrupted by
    close/cancel. Refresh invalidates older generations. wait_idle/wait_closed
    are explicit non-GUI cleanup/testing barriers, not launch deadlines.
    """

    def __init__(self, *, sdk_factory="base.vkinging_sdk.VkDaqClient", sdk_options=None,
                 on_result=None, deadline=5.0, retire_timeout=.2, max_result_bytes=65536):
        for field, value in (("deadline", deadline), ("retire_timeout", retire_timeout)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if type(max_result_bytes) is not int or not 1024 <= max_result_bytes <= 65536:
            raise ValueError("max_result_bytes must be an integer from 1024 through 65536")
        if (type(sdk_factory) is not str or "." not in sdk_factory or
                not all(part.isidentifier() for part in sdk_factory.split("."))):
            raise ValueError("sdk_factory must be an importable dotted identifier")
        if on_result is not None and not callable(on_result):
            raise ValueError("on_result must be callable or None")
        options = {} if sdk_options is None else sdk_options
        if type(options) is not dict or any(
                type(key) is not str or type(value) not in (str, int, float, bool, type(None))
                for key, value in options.items()):
            raise ValueError("sdk_options must contain only string keys and JSON scalars")
        options_json = json.dumps(options, ensure_ascii=True, allow_nan=False)
        if len(options_json) > 4096:
            raise ValueError("sdk_options exceeds byte limit")
        self._factory = sdk_factory
        self._options = options_json
        self._on_result = on_result
        self.deadline = deadline
        self.retire_timeout = retire_timeout
        self.max_result_bytes = max_result_bytes
        self._clock = time.monotonic
        self._context = multiprocessing.get_context("spawn")
        self._condition = threading.Condition()
        self._idle = threading.Event()
        self._idle.set()
        self._finished = threading.Event()
        self._closed = False
        self._generation = 0
        self._pending = None
        self._current = None
        self._result = None
        self._unreaped = None
        self._supervisor = threading.Thread(target=self._supervise,
                                            name="ve-discovery-supervisor", daemon=True)
        self._supervisor.start()

    def start(self):
        return self.refresh()

    def refresh(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("discovery service is closed")
            self._generation += 1
            if self._current is not None:
                self._current[2].set()
            self._pending = (self._generation, self._clock() + self.deadline, threading.Event())
            self._result = None
            self._idle.clear()
            self._condition.notify()
            return self._generation

    def cancel(self):
        with self._condition:
            self._generation += 1
            self._pending = None
            self._result = None
            if self._current is not None:
                self._current[2].set()
            elif self._unreaped is None:
                self._idle.set()
            self._condition.notify()

    def close(self):
        with self._condition:
            self._closed = True
            self._on_result = None
            self.cancel()

    def poll_result(self):
        with self._condition:
            result, self._result = self._result, None
            return result

    def wait_idle(self, timeout):
        return self._idle.wait(timeout)

    def wait_closed(self, timeout):
        return self._finished.wait(timeout)

    def _supervise(self):
        try:
            while True:
                self._reap_unconfirmed()
                with self._condition:
                    if self._unreaped is not None:
                        # Stay responsive to the latest request's deadline while
                        # retaining exactly one pending launch/helper/receiver.
                        if (self._closed or self._pending is None
                                or self._clock() < self._pending[1]):
                            self._condition.wait(.01)
                            continue
                    else:
                        self._condition.wait_for(lambda: self._closed or self._pending is not None)
                        if self._closed:
                            return
                    request, self._pending = self._pending, None
                    self._current = request
                if self._unreaped is not None:
                    event = DiscoveryEvent(request[0], "unavailable", DiscoveryResult(
                        diagnostics=("discovery deadline exceeded waiting for prior probe retirement",)),
                        None, 0, False)
                else:
                    event = self._probe(request)
                with self._condition:
                    self._current = None
                    callback = None
                    if not self._closed and request[0] == self._generation:
                        self._result = event
                        callback = self._on_result
                    if self._pending is None and self._unreaped is None:
                        self._idle.set()
                if callback is not None and not self._closed and request[0] == self._generation:
                    try:
                        callback(event)
                    except Exception:
                        # User callback boundary; retirement is independent of it.
                        logging.getLogger(__name__).exception("Discovery callback failed")
        finally:
            self._finished.set()

    def _probe(self, request):
        generation, expires, cancelled = request
        resources = _ProbeResources()
        mailbox = queue.Queue(maxsize=1)
        pid, size = None, 0
        result = DiscoveryResult(diagnostics=("discovery cancelled",))
        try:
            resources.receive, resources.send = self._context.Pipe(duplex=False)
            resources.child = self._context.Process(
                target=_discovery_worker,
                args=(resources.send, self._factory, self._options, self.max_result_bytes),
                name="ve-discovery-helper", daemon=True)
            if cancelled.is_set():
                raise RuntimeError("discovery cancelled")
            if self._clock() >= expires:
                raise TimeoutError("discovery deadline exceeded before spawn")
            resources.launcher = threading.Thread(target=resources.launch,
                                                  name="ve-discovery-launcher", daemon=True)
            resources.launcher.start()
            while resources.launcher.is_alive():
                if cancelled.is_set():
                    raise RuntimeError("discovery cancelled")
                if self._clock() >= expires:
                    raise TimeoutError("discovery deadline exceeded (launch)")
                cancelled.wait(.01)
            resources.launcher.join()
            if cancelled.is_set():
                raise RuntimeError("discovery cancelled")
            if self._clock() >= expires:
                raise TimeoutError("discovery deadline exceeded (launch)")
            if resources.launch_failure is not None:
                result = resources.launch_failure
            else:
                pid = resources.child.pid
                resources.send.close()
                resources.receiver = threading.Thread(
                    target=_receive_one, args=(resources.receive, mailbox, self.max_result_bytes),
                    name="ve-discovery-receiver", daemon=True)
                resources.receiver.start()
                while not cancelled.is_set():
                    if self._clock() >= expires:
                        result = DiscoveryResult(diagnostics=("discovery deadline exceeded (startup/probe)",))
                        break
                    try:
                        payload, error = mailbox.get(timeout=.01)
                    except queue.Empty:
                        continue
                    if error is not None:
                        raise error
                    if self._clock() >= expires:
                        result = DiscoveryResult(diagnostics=("discovery deadline exceeded (result)",))
                        break
                    size = len(payload)
                    result = _decode_result(payload, self.max_result_bytes)
                    break
        except (OSError, EOFError, ValueError, RuntimeError) as exc:
            result = _failure("discovery unavailable", exc)
        finally:
            cleanup = self._retire(resources)
            if not resources.released:
                self._unreaped = resources
            if cleanup:
                result = DiscoveryResult(diagnostics=(*result.diagnostics, *cleanup))
        return DiscoveryEvent(generation, "completed" if result.devices else "unavailable",
                              result, pid, size, self._unreaped is None)

    def _retire(self, resources):
        if resources.launcher is not None:
            if resources.launcher.is_alive():
                # start() may still be mutating Process or serializing send.
                # Do not even inspect its pid, or close either endpoint yet.
                return ["discovery launch pending; service quarantined until reaped"]
            if resources.launcher.ident is not None:
                resources.launcher.join()
            resources.launcher = None
        child, receiver = resources.child, resources.receiver
        cleanup = []
        if child is not None and child.pid is not None:
            child.join(self.retire_timeout)
            for operation in ("terminate", "kill"):
                if not child.is_alive():
                    break
                try:
                    getattr(child, operation)()
                except OSError as exc:
                    cleanup.extend(_failure(f"helper {operation}", exc).diagnostics)
                child.join(self.retire_timeout)
            if child.is_alive():
                cleanup.append("helper death unconfirmed after kill; service quarantined until reaped")
            if child.exitcode not in (None, 0):
                cleanup.append(f"helper exitcode={child.exitcode}")
        if resources.send is not None:
            resources.send.close()
            resources.send = None
        if resources.receive is not None:
            resources.receive.close()
            resources.receive = None
        if receiver is not None and receiver.ident is not None:
            receiver.join(self.retire_timeout)
            if receiver.is_alive():
                cleanup.append("result receiver not stopped; service quarantined until reaped")
        if child is not None and (child.pid is None or not child.is_alive()):
            child.close()
            resources.child = None
        if receiver is not None and not receiver.is_alive():
            resources.receiver = None
        return cleanup

    def _reap_unconfirmed(self):
        if self._unreaped is None:
            return
        resources = self._unreaped
        if resources.launcher is not None:
            if resources.launcher.is_alive():
                return
            if resources.launch_failure is not None:
                logging.getLogger(__name__).warning(
                    "Late %s", resources.launch_failure.diagnostics[0])
            cleanup = self._retire(resources)
            if cleanup:
                logging.getLogger(__name__).warning("Late discovery retirement: %s", "; ".join(cleanup))
        child, receiver = resources.child, resources.receiver
        if child is not None:
            if child.is_alive():
                try:
                    child.kill()
                except OSError:
                    # Uncertain OS retirement is already reported in the result;
                    # retain the handle and retry without permitting a new probe.
                    logging.getLogger(__name__).error("Discovery helper kill retry failed", exc_info=True)
                child.join(self.retire_timeout)
            if not child.is_alive():
                child.close()
                resources.child = None
        if receiver is not None:
            receiver.join(self.retire_timeout)
            if not receiver.is_alive():
                resources.receiver = None
        if resources.released:
            with self._condition:
                self._unreaped = None
                if self._pending is None:
                    self._idle.set()
