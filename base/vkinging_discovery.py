"""Pure discovery policy over a caller-owned SDK; no tasks or native loading."""
from collections import Counter
from dataclasses import dataclass
import re
from types import MappingProxyType

from base.vkinging_input import parse_vkinging_ain_route, validate_physical_channels
from base.vkinging_sdk import VkDaqError
from consts.vkinging_consts import VKINGING_HOSTAPI_NAME


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Verified immutable runtime snapshots and ordered diagnostic strings."""

    devices: tuple = ()
    diagnostics: tuple[str, ...] = ()


def _text(value, field, *, empty=False):
    # Identity and routing are exact SDK values, never case-folded or stripped.
    if (not isinstance(value, str) or (not value.strip() and not empty)
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
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
        channels.append(parse_vkinging_ain_route(route))
    return validate_physical_channels(channels)


def discover_vkinging_devices(sdk):
    """Discover without creating/closing an SDK, task, thread, or process.

    Native names are current routes; MachineId is the exact hardware_id.
    Duplicate names are not queried because attribute lookup is ambiguous.
    IDs are counted before channel validation so a broken duplicate still
    disqualifies its otherwise usable peer. Unknown external exceptions are
    left to the caller's worker boundary, preserving their original context.
    """
    try:
        entries = sdk.get_devices()
    except VkDaqError as exc:
        return DiscoveryResult(diagnostics=(str(exc),))
    if not isinstance(entries, (tuple, list)):
        return DiscoveryResult(diagnostics=("malformed device enumeration",))
    diagnostics, records, candidates, identities = [], [], [], []
    for entry in entries:
        try:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("malformed device address/name pair")
            address, name = entry
            _text(name, "name")
            if "/" in name or "," in name:
                raise ValueError("malformed device name")
            records.append((address, name))
        except ValueError as exc:
            diagnostics.append(str(exc))
    names = Counter(name for _, name in records)
    for address, name in records:
        if names[name] != 1:
            diagnostics.append(f"{name}: duplicate native name; not queried")
            continue
        try:
            machine_id = _text(sdk.get_device_attribute(name, "MachineId"), "MachineId")
            identities.append(machine_id)
            address = None if address is None else _text(address, "address", empty=True)
            channels = _analog_channels(sdk.get_channels(name), name)
        except (VkDaqError, ValueError) as exc:
            diagnostics.append(f"{name}: {exc}")
            continue
        # Model is display metadata, not a capability or identity requirement.
        model = None
        try:
            model = _text(sdk.get_device_attribute(name, "Model"), "Model", empty=True)
        except (VkDaqError, ValueError) as exc:
            diagnostics.append(f"{name}: optional Model: {exc}")
        candidates.append(MappingProxyType({
            "hostapi_name": VKINGING_HOSTAPI_NAME, "hardware_id": machine_id,
            "name": name, "address": address, "model": model,
            "physical_channels": channels, "max_input_channels": max(channels) + 1,
        }))
    counts = Counter(identities)
    for machine_id, count in counts.items():
        if count != 1:
            diagnostics.append(f"MachineId {machine_id}: duplicate stable identity")
    return DiscoveryResult(
        tuple(device for device in candidates if counts[device["hardware_id"]] == 1),
        tuple(diagnostics),
    )
