"""Resolve ordinary input identity without owning or refreshing native audio."""


def _identity_error(snapshot, reason):
    return ValueError(
        f"input device {snapshot['name']!r} on HostAPI {snapshot.get('hostapi_name', snapshot['hostapi'])!r}: "
        f"{reason}; reselect the input device.")


def validate_input_device(backend, snapshot, channels):
    """Check an explicit local slot; stable snapshots compare actual API names."""
    current = backend.query_devices(snapshot["index"])
    stable = "hostapi_name" in snapshot
    keys = ("name",) if stable else ("name", "hostapi")
    for key in keys:
        if current.get(key) != snapshot[key]:
            if stable:
                raise _identity_error(snapshot, f"identity changed at index {snapshot['index']}: {key}")
            details = "; ".join(
                f"expected {field}={snapshot.get(field)!r}, actual {field}={current.get(field)!r}"
                for field in ("name", "hostapi", "max_input_channels"))
            raise ValueError(
                f"input device identity changed at index {snapshot['index']}: {key}; "
                f"{details}; reselect the input device or restart the application.")
    if stable and backend.query_hostapis(current["hostapi"])["name"] != snapshot["hostapi_name"]:
        raise _identity_error(snapshot, f"HostAPI identity changed at index {snapshot['index']}")
    if max(channels) >= int(current.get("max_input_channels", 0)):
        if stable:
            raise _identity_error(snapshot, "no longer supports selected channels")
        raise ValueError("input device no longer supports selected channels")
    return {**snapshot, "hostapi": current["hostapi"]}


def resolve_input_device(backend, snapshot, channels):
    """After a safe refresh, bind one exact input identity or fail closed.

    Legacy snapshots deliberately retain strict numeric validation. The caller
    owns native queries and their failure boundary; this helper never resets it.
    """
    if "hostapi_name" not in snapshot:
        return validate_input_device(backend, snapshot, channels)
    apis = backend.query_hostapis()
    matches = [device for device in backend.query_devices()
               if device.get("name") == snapshot["name"]
               and int(device.get("max_input_channels", 0)) > 0
               and apis[device["hostapi"]]["name"] == snapshot["hostapi_name"]]
    if len(matches) != 1:
        reason = "exact input identity is missing" if not matches else "exact input identity is ambiguous"
        raise _identity_error(snapshot, reason)
    current = matches[0]
    if max(channels) >= int(current["max_input_channels"]):
        raise _identity_error(snapshot, "no longer supports selected channels")
    return {**snapshot, "index": current["index"], "hostapi": current["hostapi"]}
