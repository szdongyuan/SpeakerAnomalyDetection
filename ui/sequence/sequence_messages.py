"""Immutable commands and events shared by the sequence workflow domains."""

from __future__ import annotations

import array
import ctypes
import mmap
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import PosixPath, PurePath, PurePosixPath, PureWindowsPath, WindowsPath
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import DummyArray


_EXACT_NUMPY_INTEGER_TYPES = frozenset(
    {
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.intp,
        np.uintp,
        np.longlong,
        np.ulonglong,
    }
)
_EXACT_NUMPY_FLOAT_TYPES = frozenset(
    {np.float16, np.float32, np.float64, np.longdouble}
)
_EXACT_NUMPY_COMPLEX_TYPES = frozenset(
    {np.complex64, np.complex128, np.clongdouble}
)
_EXACT_NUMPY_SCALAR_TYPES = frozenset(
    {
        np.bool_,
        np.str_,
        np.bytes_,
        *_EXACT_NUMPY_INTEGER_TYPES,
        *_EXACT_NUMPY_FLOAT_TYPES,
        *_EXACT_NUMPY_COMPLEX_TYPES,
    }
)
_EXACT_NUMPY_DTYPE_TYPES = frozenset(
    type(np.dtype(specification))
    for specification in (
        "?",
        "i1",
        "i2",
        "i4",
        "i8",
        "u1",
        "u2",
        "u4",
        "u8",
        "f2",
        "f4",
        "f8",
        np.longdouble,
        "c8",
        "c16",
        np.clongdouble,
        "S1",
        "U1",
        "V1",
        "O",
        "M8[ns]",
        "m8[ns]",
    )
)
_EXACT_PATH_TYPES = frozenset(
    {PurePosixPath, PureWindowsPath, PosixPath, WindowsPath}
)


class _FrozenMapping(Mapping[Any, Any]):
    """Module-owned immutable mapping backed only by exact tuples."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, items: tuple[tuple[Any, Any], ...]) -> None:
        if type(items) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2
            for pair in tuple.__iter__(items)
        ):
            raise TypeError("frozen mapping storage must use exact key/value tuples")
        lookup = MappingProxyType(dict(items))
        if len(lookup) != len(items):
            raise ValueError("frozen mappings cannot contain duplicate keys")
        object.__setattr__(self, "_items", items)
        object.__setattr__(self, "_lookup", lookup)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("frozen mappings are immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("frozen mappings are immutable")

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_items"))

    def __iter__(self) -> Iterator[Any]:
        items = object.__getattribute__(self, "_items")
        keys = tuple(tuple.__getitem__(pair, 0) for pair in tuple.__iter__(items))
        return tuple.__iter__(keys)

    def __getitem__(self, key: Any) -> Any:
        lookup = object.__getattribute__(self, "_lookup")
        return MappingProxyType.__getitem__(lookup, key)

    def __eq__(self, other: Any) -> bool:
        lookup = object.__getattribute__(self, "_lookup")
        if type(other) is _FrozenMapping:
            other_lookup = object.__getattribute__(other, "_lookup")
            return MappingProxyType.__eq__(lookup, other_lookup)
        if type(other) is dict:
            return MappingProxyType.__eq__(lookup, other)
        return NotImplemented

    def __repr__(self) -> str:
        lookup = object.__getattribute__(self, "_lookup")
        return f"_FrozenMapping({dict(lookup)!r})"


class _SealedMessage:
    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        direct_bases = type.__getattribute__(cls, "__bases__")
        if direct_bases != (_SealedMessage,):
            raise TypeError("sequence message classes are sealed against subclassing")
        super().__init_subclass__(**kwargs)


def _trusted_type_mro(value: Any) -> tuple[type, ...]:
    return type.__getattribute__(type(value), "__mro__")


def _trusted_type_name(value: Any) -> str:
    return type.__getattribute__(type(value), "__name__")


def _type_inherits_from(value: Any, base_type: type) -> bool:
    return base_type in _trusted_type_mro(value)


def _is_dataclass_instance(value: Any) -> bool:
    for base in _trusted_type_mro(value):
        namespace = type.__getattribute__(base, "__dict__")
        if MappingProxyType.__contains__(namespace, "__dataclass_fields__"):
            return True
    return False


def _is_enum_instance(value: Any) -> bool:
    return Enum in _trusted_type_mro(value)


def _reject_behavioral_direct_field(field_name: str, value: Any) -> None:
    if _is_enum_instance(value) or _is_dataclass_instance(value):
        raise ValueError(f"{field_name} must use a plain data value")


def _exact_text(field_name: str, value: Any, *, non_empty: bool) -> str:
    _reject_behavioral_direct_field(field_name, value)
    if type(value) is str:
        normalized = value
    elif type(value) is np.str_:
        normalized = np.generic.item(value)
    else:
        raise ValueError(f"{field_name} must be a string")
    if non_empty and not normalized.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _require_generation(field_name: str, value: Any) -> int:
    return _exact_integer(field_name, value, minimum=0)


def _exact_integer(field_name: str, value: Any, *, minimum: int) -> int:
    _reject_behavioral_direct_field(field_name, value)
    if type(value) is int:
        normalized = value
    elif type(value) in _EXACT_NUMPY_INTEGER_TYPES:
        normalized = np.generic.item(value)
    else:
        raise ValueError(f"{field_name} must be an integer")
    if normalized < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return normalized


def _exact_boolean(field_name: str, value: Any) -> bool:
    _reject_behavioral_direct_field(field_name, value)
    if type(value) is bool:
        return value
    if type(value) is np.bool_:
        return np.generic.item(value)
    raise ValueError(f"{field_name} must be a boolean")


def _channel_order(field_name: str, value: Any, *, allow_empty: bool) -> tuple[int, ...]:
    _reject_behavioral_direct_field(field_name, value)
    if type(value) is tuple:
        supplied = tuple.__iter__(value)
    elif type(value) is list:
        supplied = list.__iter__(value)
    else:
        raise ValueError(
            f"{field_name} must be an ordered collection of channel indices"
        )
    normalized = tuple(
        _exact_integer(f"{field_name}[{position}]", channel, minimum=0)
        for position, channel in enumerate(supplied)
    )
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must contain at least one channel")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate channels")
    return normalized


def _validate_numpy_dtype(
    value: np.dtype, *, numeric_only: bool = False
) -> np.dtype:
    if type(value) not in _EXACT_NUMPY_DTYPE_TYPES:
        raise TypeError("NumPy payload dtype must use an exact supported dtype")
    if (
        value.hasobject
        or value.metadata is not None
        or value.fields is not None
        or value.names is not None
        or value.subdtype is not None
    ):
        raise TypeError(
            "NumPy payload dtypes must be simple and contain no objects or metadata"
        )
    if numeric_only and value.kind not in "iufc":
        raise TypeError("NumPy array payloads must have a numeric dtype")
    return value


def _copy_numpy_dtype(value: np.dtype, *, numeric_only: bool = False) -> np.dtype:
    _validate_numpy_dtype(value, numeric_only=numeric_only)
    return np.dtype(value.str)


class _ImmutableArrayFlags:
    __slots__ = ("_values",)

    def __init__(self, flags: Any) -> None:
        object.__setattr__(
            self,
            "_values",
            (
                bool(flags.c_contiguous),
                bool(flags.f_contiguous),
                bool(flags.owndata),
                bool(flags.writeable),
                bool(flags.aligned),
                bool(flags.writebackifcopy),
            ),
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("published array flags are immutable")

    @property
    def c_contiguous(self) -> bool:
        return self._values[0]

    @property
    def contiguous(self) -> bool:
        return self.c_contiguous

    @property
    def f_contiguous(self) -> bool:
        return self._values[1]

    @property
    def fortran(self) -> bool:
        return self.f_contiguous

    @property
    def owndata(self) -> bool:
        return self._values[2]

    @property
    def writeable(self) -> bool:
        return self._values[3]

    @property
    def aligned(self) -> bool:
        return self._values[4]

    @property
    def writebackifcopy(self) -> bool:
        return self._values[5]

    @property
    def behaved(self) -> bool:
        return self.aligned and self.writeable

    @property
    def carray(self) -> bool:
        return self.c_contiguous and self.behaved

    @property
    def farray(self) -> bool:
        return self.f_contiguous and not self.c_contiguous and self.behaved

    @property
    def fnc(self) -> bool:
        return self.f_contiguous and not self.c_contiguous

    @property
    def forc(self) -> bool:
        return self.c_contiguous or self.f_contiguous

    @property
    def num(self) -> int:
        return (
            int(self.c_contiguous)
            | (int(self.f_contiguous) << 1)
            | (int(self.owndata) << 2)
            | (int(self.aligned) << 8)
            | (int(self.writeable) << 10)
            | (int(self.writebackifcopy) << 13)
        )

    def __getitem__(self, key: str) -> bool:
        aliases = {
            "C": self.c_contiguous,
            "CONTIGUOUS": self.c_contiguous,
            "C_CONTIGUOUS": self.c_contiguous,
            "F": self.f_contiguous,
            "FORTRAN": self.f_contiguous,
            "F_CONTIGUOUS": self.f_contiguous,
            "O": self.owndata,
            "OWNDATA": self.owndata,
            "W": self.writeable,
            "WRITEABLE": self.writeable,
            "A": self.aligned,
            "ALIGNED": self.aligned,
            "X": self.writebackifcopy,
            "WRITEBACKIFCOPY": self.writebackifcopy,
            "FNC": self.fnc,
            "FORC": self.forc,
            "B": self.behaved,
            "BEHAVED": self.behaved,
            "CA": self.carray,
            "CARRAY": self.carray,
            "FA": self.farray,
            "FARRAY": self.farray,
        }
        try:
            return aliases[key]
        except (KeyError, TypeError) as error:
            raise KeyError("Unknown flag") from error

    def __repr__(self) -> str:
        return (
            f"  C_CONTIGUOUS : {self.c_contiguous}\n"
            f"  F_CONTIGUOUS : {self.f_contiguous}\n"
            f"  OWNDATA : {self.owndata}\n"
            f"  WRITEABLE : {self.writeable}\n"
            f"  ALIGNED : {self.aligned}\n"
            f"  WRITEBACKIFCOPY : {self.writebackifcopy}\n"
        )


class _ImmutablePayloadArray(np.ndarray):
    __slots__ = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return np.ndarray.shape.__get__(self)

    @shape.setter
    def shape(self, _value: Any) -> None:
        raise AttributeError("published array shape is immutable")

    @property
    def strides(self) -> tuple[int, ...]:
        return np.ndarray.strides.__get__(self)

    @strides.setter
    def strides(self, _value: Any) -> None:
        raise AttributeError("published array strides are immutable")

    @property
    def dtype(self) -> np.dtype:
        internal_dtype = np.ndarray.dtype.__get__(self)
        return _copy_numpy_dtype(internal_dtype)

    @dtype.setter
    def dtype(self, _value: Any) -> None:
        raise AttributeError("published array dtype is immutable")

    @property
    def data(self) -> memoryview:
        return np.ndarray.data.__get__(self)

    @data.setter
    def data(self, _value: Any) -> None:
        raise AttributeError("published array data buffer is immutable")

    @property
    def flags(self) -> _ImmutableArrayFlags:
        return _ImmutableArrayFlags(np.ndarray.flags.__get__(self))

    def setflags(self, *_args: Any, **_kwargs: Any) -> None:
        raise ValueError("published array flags are immutable")

    def __setstate__(self, _state: Any) -> None:
        raise TypeError("published array deserialization is not supported")

    def __reduce__(self) -> Any:
        raise TypeError("published array pickling is not supported")

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("published array pickling is not supported")

    def resize(self, *_args: Any, **_kwargs: Any) -> None:
        raise AttributeError("published array shape is immutable")


def _numpy_array_address(value: np.ndarray) -> int:
    interface = np.ndarray.__array_interface__.__get__(value)
    return int(interface["data"][0])


def _numpy_array_nbytes(value: np.ndarray) -> int:
    item_count = 1
    for extent in np.ndarray.shape.__get__(value):
        item_count *= int(extent)
    return item_count * int(np.ndarray.itemsize.__get__(value))


def _numpy_addressed_byte_interval(value: np.ndarray) -> tuple[int, int] | None:
    shape = tuple(np.ndarray.shape.__get__(value))
    if any(extent == 0 for extent in shape):
        return None
    strides = tuple(np.ndarray.strides.__get__(value))
    lower_offset = 0
    upper_offset = 0
    for extent, stride in zip(shape, strides):
        final_offset = (int(extent) - 1) * int(stride)
        lower_offset += min(0, final_offset)
        upper_offset += max(0, final_offset)
    address = _numpy_array_address(value)
    return (
        address + lower_offset,
        address + upper_offset + int(np.ndarray.itemsize.__get__(value)),
    )


def _ctypes_array_base_descriptor(value: ctypes.Array, name: str) -> Any:
    for base in type.__getattribute__(ctypes.Array, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if MappingProxyType.__contains__(namespace, name):
            descriptor = MappingProxyType.__getitem__(namespace, name)
            return descriptor.__get__(value, type(value))
    raise TypeError("ctypes array provenance descriptor is unavailable")


def _validate_numpy_buffer_provenance(buffer: Any) -> None:
    exporter = buffer
    visited: set[int] = set()
    while type(exporter) is memoryview:
        identity = id(exporter)
        if identity in visited:
            raise TypeError("NumPy payload backing allocation cannot be proven")
        visited.add(identity)
        exporter = exporter.obj

    if type(exporter) in (bytes, bytearray, array.array, mmap.mmap):
        return
    if isinstance(exporter, ctypes.Array):
        needs_free = _ctypes_array_base_descriptor(exporter, "_b_needsfree_")
        base = _ctypes_array_base_descriptor(exporter, "_b_base_")
        if type(needs_free) is int and needs_free == 1 and base is None:
            return
        raise TypeError("NumPy payload backing allocation cannot be proven")
    if type(exporter) in (np.ndarray, _ImmutablePayloadArray):
        _validate_numpy_array_allocation(exporter)
        return
    if isinstance(exporter, np.ndarray):
        raise TypeError("NumPy payload backing allocation cannot use array subclasses")
    raise TypeError("NumPy payload backing allocation cannot be proven")


def _numpy_buffer_byte_interval(buffer: Any) -> tuple[int, int]:
    _validate_numpy_buffer_provenance(buffer)
    try:
        buffer_view = memoryview(buffer)
        byte_view = np.frombuffer(buffer_view, dtype=np.uint8)
    except (BufferError, TypeError, ValueError) as error:
        raise TypeError("NumPy payload backing allocation cannot be proven") from error
    address = _numpy_array_address(byte_view)
    return address, address + _numpy_array_nbytes(byte_view)


def _numpy_backing_allocation_interval(value: np.ndarray) -> tuple[int, int]:
    current = value
    visited: set[int] = set()
    while True:
        identity = id(current)
        if identity in visited:
            raise TypeError("NumPy payload backing allocation cannot be proven")
        visited.add(identity)

        base = np.ndarray.base.__get__(current)
        if type(base) in (np.ndarray, _ImmutablePayloadArray):
            current = base
            continue
        if isinstance(base, np.ndarray):
            raise TypeError("NumPy payload backing allocation cannot use array subclasses")
        if type(base) is DummyArray:
            nested_base = object.__getattribute__(base, "base")
            if type(nested_base) not in (np.ndarray, _ImmutablePayloadArray):
                raise TypeError("NumPy payload backing allocation cannot be proven")
            current = nested_base
            continue
        if base is None:
            if not np.ndarray.flags.__get__(current).owndata:
                raise TypeError("NumPy payload backing allocation cannot be proven")
            address = _numpy_array_address(current)
            return address, address + _numpy_array_nbytes(current)
        return _numpy_buffer_byte_interval(base)


def _validate_numpy_array_allocation(value: np.ndarray) -> None:
    addressed_interval = _numpy_addressed_byte_interval(value)
    if addressed_interval is None:
        return
    allocation_start, allocation_stop = _numpy_backing_allocation_interval(value)
    addressed_start, addressed_stop = addressed_interval
    if addressed_start < allocation_start or addressed_stop > allocation_stop:
        raise TypeError("NumPy payload view exceeds its backing allocation")


def _build_immutable_numpy_array(
    *, shape: tuple[int, ...], dtype: np.dtype, immutable_bytes: bytes
) -> np.ndarray:
    if type(immutable_bytes) is not bytes:
        raise RuntimeError("published NumPy payload must use exact immutable bytes")
    frozen = np.ndarray.__new__(
        _ImmutablePayloadArray,
        shape=shape,
        dtype=dtype,
        buffer=immutable_bytes,
        order="C",
    )
    if type(np.ndarray.base.__get__(frozen)) is not bytes:
        raise RuntimeError("published NumPy payload must have an immutable bytes buffer")
    if np.ndarray.flags.__get__(frozen).writeable:
        raise RuntimeError("published NumPy payload must be read-only")
    return frozen


def _copy_numpy_array_to_immutable(
    value: np.ndarray, *, validate_provenance: bool
) -> np.ndarray:
    source_dtype = np.ndarray.dtype.__get__(value)
    source_shape = tuple(np.ndarray.shape.__get__(value))
    dtype = _copy_numpy_dtype(source_dtype, numeric_only=True)
    if validate_provenance:
        _validate_numpy_array_allocation(value)
    immutable_bytes = np.ndarray.tobytes(value, order="C")
    return _build_immutable_numpy_array(
        shape=source_shape, dtype=dtype, immutable_bytes=immutable_bytes
    )


def _immutable_numpy_allocation_root(
    value: _ImmutablePayloadArray,
) -> _ImmutablePayloadArray:
    current: np.ndarray = value
    visited: set[int] = set()
    while True:
        identity = id(current)
        if identity in visited:
            raise TypeError("immutable NumPy payload allocation cannot be proven")
        visited.add(identity)

        base = np.ndarray.base.__get__(current)
        if type(base) in (np.ndarray, _ImmutablePayloadArray):
            current = base
            continue
        if isinstance(base, np.ndarray):
            raise TypeError("immutable NumPy payload cannot use array subclasses")
        if type(base) is DummyArray:
            nested_base = object.__getattribute__(base, "base")
            if type(nested_base) not in (np.ndarray, _ImmutablePayloadArray):
                raise TypeError("immutable NumPy payload allocation cannot be proven")
            current = nested_base
            continue
        if type(base) is not bytes or type(current) is not _ImmutablePayloadArray:
            raise TypeError("immutable NumPy payload allocation cannot be proven")

        flags = np.ndarray.flags.__get__(current)
        if flags.writeable or not flags.c_contiguous:
            raise TypeError(
                "immutable NumPy payload root must be C-contiguous and bytes-backed"
            )
        return current


def _validate_immutable_numpy_array(
    value: _ImmutablePayloadArray,
) -> _ImmutablePayloadArray:
    dtype = np.ndarray.dtype.__get__(value)
    _validate_numpy_dtype(dtype, numeric_only=True)
    root = _immutable_numpy_allocation_root(value)
    _validate_numpy_array_allocation(value)
    return root


def _freeze_numpy_array(
    value: np.ndarray, *, reuse_immutable: bool = False
) -> np.ndarray:
    if type(value) is _ImmutablePayloadArray:
        root = _validate_immutable_numpy_array(value)
        if reuse_immutable and value is root:
            return value
        return _copy_numpy_array_to_immutable(value, validate_provenance=True)
    if type(value) is not np.ndarray:
        raise TypeError("NumPy payload arrays must use an exact NumPy ndarray")
    return _copy_numpy_array_to_immutable(value, validate_provenance=True)


def _detach_immutable_numpy_array_shell(
    value: _ImmutablePayloadArray,
) -> np.ndarray:
    """Rebuild mutable ndarray metadata while sharing only proven frozen bytes."""

    root = _validate_immutable_numpy_array(value)
    if value is not root:
        # A strided/view shell cannot be represented as the canonical C-order
        # payload without either retaining owner metadata or copying its logical
        # bytes.  Copy only this view case; exact roots share immutable bytes.
        return _copy_numpy_array_to_immutable(value, validate_provenance=True)
    immutable_bytes = np.ndarray.base.__get__(root)
    if type(immutable_bytes) is not bytes:
        raise TypeError("immutable NumPy payload root must be bytes-backed")
    return _build_immutable_numpy_array(
        shape=tuple(np.ndarray.shape.__get__(value)),
        dtype=_copy_numpy_dtype(
            np.ndarray.dtype.__get__(value),
            numeric_only=True,
        ),
        immutable_bytes=immutable_bytes,
    )


def _freeze_callback_numpy_array(value: np.ndarray) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError("callback arrays must use an exact NumPy ndarray")
    return _copy_numpy_array_to_immutable(value, validate_provenance=False)


def _freeze_numpy_scalar(value: np.generic) -> Any:
    scalar_type = type(value)
    if scalar_type not in _EXACT_NUMPY_SCALAR_TYPES:
        raise TypeError("NumPy scalar payload must use an exact supported NumPy scalar")
    _validate_numpy_dtype(np.generic.dtype.__get__(value))
    normalized = np.generic.item(value)
    if scalar_type is np.bool_:
        return normalized
    if scalar_type in _EXACT_NUMPY_INTEGER_TYPES:
        return normalized
    if scalar_type in _EXACT_NUMPY_FLOAT_TYPES:
        return float(normalized)
    if scalar_type in _EXACT_NUMPY_COMPLEX_TYPES:
        return complex(normalized)
    if scalar_type is np.str_:
        return normalized
    if scalar_type is np.bytes_:
        return normalized
    raise TypeError(
        f"unsupported NumPy scalar message payload: {_trusted_type_name(value)}"
    )


def _freeze_path(value: PurePath) -> PurePath:
    path_type = type(value)
    if path_type not in _EXACT_PATH_TYPES:
        raise TypeError("path payload must use an exact supported PurePath type")
    parts = tuple(str.__str__(part) for part in PurePath.parts.__get__(value))
    return path_type(*parts)


def _freeze_timedelta_components(value: timedelta) -> timedelta:
    return timedelta(
        days=timedelta.days.__get__(value),
        seconds=timedelta.seconds.__get__(value),
        microseconds=timedelta.microseconds.__get__(value),
    )


def _freeze_timezone(value: timezone) -> timezone:
    source_offset = timezone.utcoffset(value, None)
    offset = _freeze_timedelta_components(source_offset)
    source_name = timezone.tzname(value, None)
    if str not in _trusted_type_mro(source_name):
        raise TypeError("timezone names must be strings")
    name = str.__str__(source_name)
    return timezone(offset, name)


def _freeze_temporal_timezone(value: datetime | time) -> timezone | None:
    if type(value) is datetime:
        source_timezone = datetime.tzinfo.__get__(value)
    else:
        source_timezone = time.tzinfo.__get__(value)
    if source_timezone is None:
        return None
    if type(source_timezone) is not timezone:
        raise TypeError("temporal payload tzinfo must be an exact datetime.timezone")
    return _freeze_timezone(source_timezone)


def _freeze_date(value: date) -> date:
    return date(
        date.year.__get__(value),
        date.month.__get__(value),
        date.day.__get__(value),
    )


def _freeze_datetime(value: datetime) -> datetime:
    return datetime(
        datetime.year.__get__(value),
        datetime.month.__get__(value),
        datetime.day.__get__(value),
        datetime.hour.__get__(value),
        datetime.minute.__get__(value),
        datetime.second.__get__(value),
        datetime.microsecond.__get__(value),
        tzinfo=_freeze_temporal_timezone(value),
        fold=datetime.fold.__get__(value),
    )


def _freeze_time(value: time) -> time:
    return time(
        time.hour.__get__(value),
        time.minute.__get__(value),
        time.second.__get__(value),
        time.microsecond.__get__(value),
        tzinfo=_freeze_temporal_timezone(value),
        fold=time.fold.__get__(value),
    )


def _freeze_payload(
    value: Any,
    active_path: set[int] | None = None,
    *,
    array_memo: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    array_occurrences: list[int] | None = None,
    detach_configuration_snapshots: bool = False,
    detach_immutable_arrays: bool = False,
) -> Any:
    """Freeze the deliberately small, data-only message payload protocol."""

    if active_path is None:
        active_path = set()

    if type(value) is ConfigurationSnapshot:
        if detach_configuration_snapshots:
            return _detach_configuration_snapshot(
                value,
                active_path,
                array_memo=array_memo,
                array_occurrences=array_occurrences,
                detach_immutable_arrays=detach_immutable_arrays,
            )
        return value

    if _is_dataclass_instance(value):
        raise TypeError(
            "unsupported dataclass message payload; use an explicit snapshot type"
        )

    if _is_enum_instance(value):
        raise TypeError(
            "Enum message payloads are unsupported; publish a primitive stable token"
        )

    value_type = type(value)

    if value_type in (np.ndarray, _ImmutablePayloadArray):
        identity = id(value)
        if array_occurrences is not None:
            array_occurrences.append(identity)
        if array_memo is not None:
            cached = array_memo.get(identity)
            if cached is not None and cached[0] is value:
                return cached[1]
        if value_type is _ImmutablePayloadArray and detach_immutable_arrays:
            frozen_array = _detach_immutable_numpy_array_shell(value)
        else:
            frozen_array = _freeze_numpy_array(value, reuse_immutable=True)
        if array_memo is not None:
            array_memo[identity] = (value, frozen_array)
        return frozen_array
    if _type_inherits_from(value, np.ndarray):
        raise TypeError("NumPy payload arrays must use an exact NumPy ndarray")

    if value_type in _EXACT_NUMPY_SCALAR_TYPES:
        return _freeze_numpy_scalar(value)
    if _type_inherits_from(value, np.generic):
        raise TypeError("NumPy scalar payload must use an exact supported NumPy scalar")

    if value_type in _EXACT_NUMPY_DTYPE_TYPES:
        return _copy_numpy_dtype(value)
    if _type_inherits_from(value, np.dtype):
        raise TypeError("NumPy dtype payload must use an exact supported dtype")

    if value_type in _EXACT_PATH_TYPES:
        return _freeze_path(value)

    if value_type in (dict, _FrozenMapping):
        identity = id(value)
        if identity in active_path:
            raise TypeError("cyclic message payloads are not supported")
        active_path.add(identity)
        try:
            if value_type is dict:
                items = dict.items(value)
            else:
                stored_items = object.__getattribute__(value, "_items")
                if type(stored_items) is not tuple or any(
                    type(pair) is not tuple or tuple.__len__(pair) != 2
                    for pair in tuple.__iter__(stored_items)
                ):
                    raise TypeError(
                        "frozen mapping storage must use exact key/value tuples"
                    )
                items = tuple.__iter__(stored_items)
            return _FrozenMapping(
                tuple(
                    (
                        _freeze_nested_payload(
                            key,
                            active_path,
                            array_memo,
                            array_occurrences,
                            detach_configuration_snapshots,
                            detach_immutable_arrays,
                        ),
                        _freeze_nested_payload(
                            item,
                            active_path,
                            array_memo,
                            array_occurrences,
                            detach_configuration_snapshots,
                            detach_immutable_arrays,
                        ),
                    )
                    for key, item in items
                )
            )
        finally:
            active_path.remove(identity)
    if value_type is MappingProxyType:
        raise TypeError("external mapping proxy payloads are unsupported")
    if value_type in (list, tuple):
        identity = id(value)
        if identity in active_path:
            raise TypeError("cyclic message payloads are not supported")
        active_path.add(identity)
        try:
            supplied = (
                list.__iter__(value) if value_type is list else tuple.__iter__(value)
            )
            return tuple(
                _freeze_nested_payload(
                    item,
                    active_path,
                    array_memo,
                    array_occurrences,
                    detach_configuration_snapshots,
                    detach_immutable_arrays,
                )
                for item in supplied
            )
        finally:
            active_path.remove(identity)
    if value_type in (set, frozenset):
        identity = id(value)
        if identity in active_path:
            raise TypeError("cyclic message payloads are not supported")
        active_path.add(identity)
        try:
            supplied = (
                set.__iter__(value)
                if value_type is set
                else frozenset.__iter__(value)
            )
            return frozenset(
                _freeze_nested_payload(
                    item,
                    active_path,
                    array_memo,
                    array_occurrences,
                    detach_configuration_snapshots,
                    detach_immutable_arrays,
                )
                for item in supplied
            )
        finally:
            active_path.remove(identity)

    if value_type is type(None):
        return None
    if value_type in (str, bytes, bool, int, float, complex):
        return value
    if value_type is Decimal:
        return value
    if value_type is date:
        return _freeze_date(value)
    if value_type is datetime:
        return _freeze_datetime(value)
    if value_type is time:
        return _freeze_time(value)
    if value_type is timedelta:
        return _freeze_timedelta_components(value)
    if value_type is timezone:
        return _freeze_timezone(value)

    raise TypeError(
        f"unsupported mutable message payload type: {_trusted_type_name(value)}"
    )


def _freeze_nested_payload(
    value: Any,
    active_path: set[int],
    array_memo: dict[int, tuple[np.ndarray, np.ndarray]] | None,
    array_occurrences: list[int] | None,
    detach_configuration_snapshots: bool,
    detach_immutable_arrays: bool,
) -> Any:
    if (
        array_memo is None
        and array_occurrences is None
        and not detach_configuration_snapshots
        and not detach_immutable_arrays
    ):
        return _freeze_payload(value, active_path)
    return _freeze_payload(
        value,
        active_path,
        array_memo=array_memo,
        array_occurrences=array_occurrences,
        detach_configuration_snapshots=detach_configuration_snapshots,
        detach_immutable_arrays=detach_immutable_arrays,
    )


def _freeze_fields(message: Any, *field_names: str) -> None:
    for field_name in field_names:
        object.__setattr__(
            message,
            field_name,
            _freeze_payload(object.__getattribute__(message, field_name)),
        )


def _normalize_text_fields(
    message: Any, *field_names: str, non_empty: bool = False
) -> None:
    for field_name in field_names:
        object.__setattr__(
            message,
            field_name,
            _exact_text(
                field_name,
                object.__getattribute__(message, field_name),
                non_empty=non_empty,
            ),
        )


def _validate_identifiers(message: Any, *field_names: str) -> None:
    _normalize_text_fields(message, *field_names, non_empty=True)


def _require_ndarray(field_name: str, value: Any) -> None:
    _reject_behavioral_direct_field(field_name, value)
    if type(value) not in (np.ndarray, _ImmutablePayloadArray):
        raise TypeError(f"{field_name} must be an exact NumPy ndarray")


def _require_callback_ndarray(field_name: str, value: Any) -> None:
    _reject_behavioral_direct_field(field_name, value)
    if type(value) is not np.ndarray:
        raise TypeError(f"{field_name} must be an exact NumPy ndarray")


def _validate_mono_shape(value: np.ndarray, *, frame_count: int) -> None:
    shape = tuple(np.ndarray.shape.__get__(value))
    if not (
        (len(shape) == 1 and shape[0] == frame_count)
        or (len(shape) == 2 and shape == (frame_count, 1))
    ):
        raise ValueError(
            "mono shape must be (frames,) or (frames, 1) with the same number "
            "of frames as multi"
        )


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot(_SealedMessage):
    sequence_config: Any
    analysis_config: Any
    mic: Any = None
    speaker: Any = None
    mic_channels: tuple[int, ...] = ()
    using_config_path: Any = None
    streaming_stimulus_data: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mic_channels",
            _channel_order("mic_channels", self.mic_channels, allow_empty=True),
        )
        _freeze_fields(
            self,
            "sequence_config",
            "analysis_config",
            "mic",
            "speaker",
            "using_config_path",
            "streaming_stimulus_data",
        )


_CONFIGURATION_SNAPSHOT_PAYLOAD_FIELDS = (
    "sequence_config",
    "analysis_config",
    "mic",
    "speaker",
    "using_config_path",
    "streaming_stimulus_data",
)


def _detach_configuration_snapshot(
    value: ConfigurationSnapshot,
    active_path: set[int],
    *,
    array_memo: dict[int, tuple[np.ndarray, np.ndarray]] | None,
    array_occurrences: list[int] | None,
    detach_immutable_arrays: bool,
) -> ConfigurationSnapshot:
    """Strictly rebuild one configuration for a cross-owner handoff.

    Ordinary in-process message composition keeps reusing a canonical
    ``ConfigurationSnapshot`` for compatibility.  A recording admission crosses
    policy/worker ownership, so it opts into this stronger boundary: every slot
    is read without user hooks, the complete payload subtree is frozen with one
    traversal context, and the replaceable snapshot shell is rebuilt.
    """

    identity = id(value)
    if identity in active_path:
        raise TypeError("cyclic message payloads are not supported")
    active_path.add(identity)
    try:
        mic_channels = _channel_order(
            "mic_channels",
            object.__getattribute__(value, "mic_channels"),
            allow_empty=True,
        )
        raw_fields = tuple(
            object.__getattribute__(value, name)
            for name in _CONFIGURATION_SNAPSHOT_PAYLOAD_FIELDS
        )
        frozen_fields = _freeze_payload(
            raw_fields,
            active_path,
            array_memo=array_memo,
            array_occurrences=array_occurrences,
            detach_configuration_snapshots=True,
            detach_immutable_arrays=detach_immutable_arrays,
        )
    finally:
        active_path.remove(identity)

    rebuilt = object.__new__(ConfigurationSnapshot)
    for name, frozen in zip(
        _CONFIGURATION_SNAPSHOT_PAYLOAD_FIELDS,
        tuple.__iter__(frozen_fields),
    ):
        object.__setattr__(rebuilt, name, frozen)
    object.__setattr__(rebuilt, "mic_channels", mic_channels)
    return rebuilt


# Workflow and trigger commands


@dataclass(frozen=True, slots=True)
class StartTestRequested(_SealedMessage):
    command_id: str
    source: str
    label: str
    skip_sn_regex_validation: bool
    configuration_generation: int

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "source")
        _normalize_text_fields(self, "label")
        object.__setattr__(
            self,
            "skip_sn_regex_validation",
            _exact_boolean("skip_sn_regex_validation", self.skip_sn_regex_validation),
        )
        object.__setattr__(
            self,
            "configuration_generation",
            _require_generation("configuration_generation", self.configuration_generation),
        )


@dataclass(frozen=True, slots=True)
class ReplayRequested(_SealedMessage):
    command_id: str
    source: str
    record_id: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "source", "record_id")


@dataclass(frozen=True, slots=True)
class ImportAudioRequested(_SealedMessage):
    command_id: str
    mode: str
    selected_path: Any = None

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id")
        _normalize_text_fields(self, "mode")
        _freeze_fields(self, "selected_path")


@dataclass(frozen=True, slots=True)
class BarcodeCommitted(_SealedMessage):
    command_id: str
    source: str
    barcode: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "source")
        _normalize_text_fields(self, "barcode")


@dataclass(frozen=True, slots=True)
class ManualLabelRequested(_SealedMessage):
    command_id: str
    record_id: str
    label: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "record_id")
        _normalize_text_fields(self, "label")


@dataclass(frozen=True, slots=True)
class ManualAnalysisRequested(_SealedMessage):
    command_id: str
    record_id: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "record_id")


# Commands admitted to domain controllers


@dataclass(frozen=True, slots=True)
class BeginRecordingRequested(_SealedMessage):
    command_id: str
    session_id: str
    replay: bool
    session_snapshot: Any

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "session_id")
        object.__setattr__(self, "replay", _exact_boolean("replay", self.replay))
        _freeze_fields(self, "session_snapshot")


@dataclass(frozen=True, slots=True)
class RecordingMarkActionRequested(_SealedMessage):
    """Request one generation-bound Recording-owned mark-mode cleanup."""

    command_id: str
    workflow_generation: int

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )


@dataclass(frozen=True, slots=True)
class LoadImportedAudioRequested(_SealedMessage):
    command_id: str
    import_id: str
    mode: str
    selected_path: Any
    configuration_snapshot: Any
    workflow_generation: int = 0

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "import_id")
        _normalize_text_fields(self, "mode")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )
        _freeze_fields(self, "selected_path", "configuration_snapshot")


@dataclass(frozen=True, slots=True)
class AnalysisRequested(_SealedMessage):
    analysis_id: str
    source_id: str
    recording_snapshot: Any
    configuration_snapshot: Any
    automatic: bool
    workflow_generation: int = 0

    def __post_init__(self) -> None:
        _validate_identifiers(self, "analysis_id", "source_id")
        object.__setattr__(self, "automatic", _exact_boolean("automatic", self.automatic))
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )
        _freeze_fields(self, "recording_snapshot", "configuration_snapshot")


@dataclass(frozen=True, slots=True)
class ExportRequested(_SealedMessage):
    job_id: str
    record_id: str
    result_snapshot: Any
    target_configurations: tuple[Any, ...]

    def __post_init__(self) -> None:
        _validate_identifiers(self, "job_id", "record_id")
        _freeze_fields(self, "result_snapshot", "target_configurations")


@dataclass(frozen=True, slots=True)
class PrepareAnalysisExportRequested(_SealedMessage):
    request_id: str
    analysis_id: str
    source_id: str
    record_id: str
    workflow_generation: int
    result_snapshot: Any
    analysis_configuration: Any = None

    def __post_init__(self) -> None:
        _validate_identifiers(
            self, "request_id", "analysis_id", "source_id", "record_id"
        )
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )
        _freeze_fields(self, "result_snapshot", "analysis_configuration")


@dataclass(frozen=True, slots=True)
class PrepareManualLabelExportRequested(_SealedMessage):
    request_id: str
    command_id: str
    record_id: str
    label: str
    workflow_generation: int

    def __post_init__(self) -> None:
        _validate_identifiers(self, "request_id", "command_id", "record_id")
        _normalize_text_fields(self, "label")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )


@dataclass(frozen=True, slots=True)
class CancelExportPreparationRequested(_SealedMessage):
    request_id: str
    workflow_generation: int
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "request_id")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )
        _normalize_text_fields(self, "reason")


@dataclass(frozen=True, slots=True)
class CancelWorkflowRequested(_SealedMessage):
    command_id: str
    workflow_generation: int
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id")
        _normalize_text_fields(self, "reason")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )


@dataclass(frozen=True, slots=True)
class CancelRecordingRequested(_SealedMessage):
    session_id: str
    workflow_generation: int
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        _normalize_text_fields(self, "reason")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )


@dataclass(frozen=True, slots=True)
class CancelImportedAudioRequested(_SealedMessage):
    import_id: str
    workflow_generation: int
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "import_id")
        _normalize_text_fields(self, "reason")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )


@dataclass(frozen=True, slots=True)
class CancelAnalysisRequested(_SealedMessage):
    analysis_id: str
    workflow_generation: int
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "analysis_id")
        _normalize_text_fields(self, "reason")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )


@dataclass(frozen=True, slots=True)
class CancelExportRequested(_SealedMessage):
    job_id: str
    workflow_generation: int
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "job_id")
        _normalize_text_fields(self, "reason")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )


@dataclass(frozen=True, slots=True)
class CommitRecordingLabelRequested(_SealedMessage):
    command_id: str
    record_id: str
    label: str
    export_outcome: Any

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "record_id")
        _normalize_text_fields(self, "label")
        _freeze_fields(self, "export_outcome")


@dataclass(frozen=True, slots=True)
class RetryExportRequested(_SealedMessage):
    job_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "job_id", "attempt_id")


@dataclass(frozen=True, slots=True)
class IgnoreExportFailureRequested(_SealedMessage):
    job_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "job_id", "attempt_id")


@dataclass(frozen=True, slots=True)
class ShutdownRequested(_SealedMessage):
    shutdown_generation: int
    has_active_workflow: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )
        object.__setattr__(
            self,
            "has_active_workflow",
            _exact_boolean("has_active_workflow", self.has_active_workflow),
        )


@dataclass(frozen=True, slots=True)
class ConfirmShutdownCancellationRequested(_SealedMessage):
    shutdown_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )


@dataclass(frozen=True, slots=True)
class AbortShutdownRequested(_SealedMessage):
    shutdown_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )


@dataclass(frozen=True, slots=True)
class RetryShutdownFlushRequested(_SealedMessage):
    shutdown_generation: int
    job_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )
        _validate_identifiers(self, "job_id", "attempt_id")


@dataclass(frozen=True, slots=True)
class IgnoreShutdownFlushFailureRequested(_SealedMessage):
    shutdown_generation: int
    job_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )
        _validate_identifiers(self, "job_id", "attempt_id")


@dataclass(frozen=True, slots=True)
class BeginShutdownFlushRequested(_SealedMessage):
    shutdown_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )


@dataclass(frozen=True, slots=True)
class ResourceLifecycleRequested(_SealedMessage):
    """One exact permanent-shutdown lifecycle operation."""

    shutdown_generation: int
    operation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )
        _normalize_text_fields(self, "operation", non_empty=True)


# Audio FIFO messages


@dataclass(frozen=True, slots=True)
class AudioBatch(_SealedMessage):
    session_id: str
    sequence_no: int
    sample_start: int
    sample_stop: int
    multi: np.ndarray
    channel_order: tuple[int, ...]
    mono: np.ndarray | None = None

    def __getitem__(self, key: str) -> np.ndarray | None:
        """Preserve the narrow legacy chunk payload lookup contract."""
        if key == "mono":
            return self.mono
        if key == "multi":
            return self.multi
        raise KeyError(key)

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        sequence_no = _exact_integer("sequence_no", self.sequence_no, minimum=0)
        sample_start = _exact_integer("sample_start", self.sample_start, minimum=0)
        sample_stop = _exact_integer("sample_stop", self.sample_stop, minimum=0)
        if sample_stop < sample_start:
            raise ValueError("sample range must be non-negative and ordered")

        _require_ndarray("multi", self.multi)
        multi_ndim = np.ndarray.ndim.__get__(self.multi)
        multi_shape = tuple(np.ndarray.shape.__get__(self.multi))
        if multi_ndim != 2:
            raise ValueError("multi must be a two-dimensional array")
        if multi_shape[0] != sample_stop - sample_start:
            raise ValueError("sample range must match the audio frame count")

        channel_order = _channel_order("channel_order", self.channel_order, allow_empty=False)
        if len(channel_order) != multi_shape[1]:
            raise ValueError("channel_order must describe every multi-array channel")

        if self.mono is not None:
            _require_ndarray("mono", self.mono)
            _validate_mono_shape(self.mono, frame_count=multi_shape[0])

        multi = _freeze_numpy_array(self.multi, reuse_immutable=True)
        mono = (
            None
            if self.mono is None
            else _freeze_numpy_array(self.mono, reuse_immutable=True)
        )

        object.__setattr__(self, "sequence_no", sequence_no)
        object.__setattr__(self, "sample_start", sample_start)
        object.__setattr__(self, "sample_stop", sample_stop)
        object.__setattr__(self, "multi", multi)
        object.__setattr__(self, "channel_order", channel_order)
        object.__setattr__(self, "mono", mono)

    @classmethod
    def from_callback(
        cls,
        *,
        session_id: str,
        sequence_no: int,
        sample_start: int,
        multi: np.ndarray,
        channel_order: tuple[int, ...],
        mono: np.ndarray | None = None,
    ) -> AudioBatch:
        normalized_session_id = _exact_text("session_id", session_id, non_empty=True)
        normalized_sequence_no = _exact_integer(
            "sequence_no", sequence_no, minimum=0
        )
        normalized_sample_start = _exact_integer(
            "sample_start", sample_start, minimum=0
        )
        normalized_channel_order = _channel_order(
            "channel_order", channel_order, allow_empty=False
        )

        _require_callback_ndarray("multi", multi)
        if mono is not None:
            _require_callback_ndarray("mono", mono)

        multi_ndim = np.ndarray.ndim.__get__(multi)
        multi_shape = tuple(np.ndarray.shape.__get__(multi))
        multi_dtype = np.ndarray.dtype.__get__(multi)
        _validate_numpy_dtype(multi_dtype, numeric_only=True)
        if multi_ndim != 2:
            raise ValueError("multi must be a two-dimensional array")
        if len(normalized_channel_order) != multi_shape[1]:
            raise ValueError("channel_order must describe every multi-array channel")

        if mono is not None:
            mono_dtype = np.ndarray.dtype.__get__(mono)
            _validate_numpy_dtype(mono_dtype, numeric_only=True)
            _validate_mono_shape(mono, frame_count=multi_shape[0])

        # sounddevice's callback arrays may be backed by transient CFFI buffers.
        # Copy each retained payload directly into its final immutable bytes backing
        # while the callback lifetime is valid. Direct message construction remains
        # subject to conservative provenance checks.
        multi_array = _freeze_callback_numpy_array(multi)
        mono_array = (
            None
            if mono is None
            else _freeze_callback_numpy_array(mono)
        )
        sample_stop = normalized_sample_start + multi_shape[0]
        return cls(
            session_id=normalized_session_id,
            sequence_no=normalized_sequence_no,
            sample_start=normalized_sample_start,
            sample_stop=sample_stop,
            multi=multi_array,
            channel_order=normalized_channel_order,
            mono=mono_array,
        )


@dataclass(frozen=True, slots=True)
class AudioCompleted(_SealedMessage):
    session_id: str
    last_sequence_no: int
    sample_count: int

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        object.__setattr__(
            self,
            "last_sequence_no",
            _exact_integer("last_sequence_no", self.last_sequence_no, minimum=-1),
        )
        object.__setattr__(
            self,
            "sample_count",
            _exact_integer("sample_count", self.sample_count, minimum=0),
        )


@dataclass(frozen=True, slots=True)
class AudioFailed(_SealedMessage):
    session_id: str
    last_sequence_no: int
    error_code: str
    message: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        _normalize_text_fields(self, "error_code", "message")
        object.__setattr__(
            self,
            "last_sequence_no",
            _exact_integer("last_sequence_no", self.last_sequence_no, minimum=-1),
        )


@dataclass(frozen=True, slots=True)
class AudioCancelled(_SealedMessage):
    session_id: str
    last_sequence_no: int
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        _normalize_text_fields(self, "reason")
        object.__setattr__(
            self,
            "last_sequence_no",
            _exact_integer("last_sequence_no", self.last_sequence_no, minimum=-1),
        )


# Domain and workflow events


@dataclass(frozen=True, slots=True)
class RecordingStarted(_SealedMessage):
    session_id: str
    session_snapshot: Any

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        _freeze_fields(self, "session_snapshot")


@dataclass(frozen=True, slots=True)
class RecordingBatchReady(_SealedMessage):
    session_id: str
    sequence_no: int
    sample_start: int
    sample_stop: int
    display: np.ndarray

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        sequence_no = _exact_integer("sequence_no", self.sequence_no, minimum=0)
        sample_start = _exact_integer("sample_start", self.sample_start, minimum=0)
        sample_stop = _exact_integer("sample_stop", self.sample_stop, minimum=0)
        if sample_stop < sample_start:
            raise ValueError("sample range must be non-negative and ordered")
        _require_ndarray("display", self.display)
        display = _freeze_payload(self.display)
        display_ndim = np.ndarray.ndim.__get__(display)
        display_shape = tuple(np.ndarray.shape.__get__(display))
        if (
            display_ndim not in (1, 2)
            or display_shape[0] != sample_stop - sample_start
        ):
            raise ValueError("display array must match the event sample range")
        object.__setattr__(self, "sequence_no", sequence_no)
        object.__setattr__(self, "sample_start", sample_start)
        object.__setattr__(self, "sample_stop", sample_stop)
        object.__setattr__(self, "display", display)


@dataclass(frozen=True, slots=True)
class RecordingCompleted(_SealedMessage):
    session_id: str
    sample_count: int
    result_snapshot: Any

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        object.__setattr__(
            self,
            "sample_count",
            _exact_integer("sample_count", self.sample_count, minimum=0),
        )
        _freeze_fields(self, "result_snapshot")


@dataclass(frozen=True, slots=True)
class RecordingFailed(_SealedMessage):
    session_id: str
    reason: str
    rollback_outcome: Any = None
    audio_committed: bool = False
    recovery_path: Any = None

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        _normalize_text_fields(self, "reason")
        object.__setattr__(
            self,
            "audio_committed",
            _exact_boolean("audio_committed", self.audio_committed),
        )
        _freeze_fields(self, "rollback_outcome", "recovery_path")


@dataclass(frozen=True, slots=True)
class RecordingCancelled(_SealedMessage):
    session_id: str
    reason: str
    rollback_outcome: Any = None

    def __post_init__(self) -> None:
        _validate_identifiers(self, "session_id")
        _normalize_text_fields(self, "reason")
        _freeze_fields(self, "rollback_outcome")


@dataclass(frozen=True, slots=True)
class ImportedAudioReady(_SealedMessage):
    import_id: str
    recording_snapshot: Any
    reference_snapshot: Any = None

    def __post_init__(self) -> None:
        _validate_identifiers(self, "import_id")
        _freeze_fields(self, "recording_snapshot", "reference_snapshot")


@dataclass(frozen=True, slots=True)
class ImportedAudioFailed(_SealedMessage):
    import_id: str
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "import_id")
        _normalize_text_fields(self, "reason")


@dataclass(frozen=True, slots=True)
class AnalysisCompleted(_SealedMessage):
    analysis_id: str
    source_id: str
    result_snapshot: Any

    def __post_init__(self) -> None:
        _validate_identifiers(self, "analysis_id", "source_id")
        _freeze_fields(self, "result_snapshot")


@dataclass(frozen=True, slots=True)
class AnalysisExportPrepared(_SealedMessage):
    request_id: str
    analysis_id: str
    source_id: str
    record_id: str
    workflow_generation: int
    result_snapshot: Any
    target_configurations: tuple[Any, ...]

    def __post_init__(self) -> None:
        _validate_identifiers(
            self, "request_id", "analysis_id", "source_id", "record_id"
        )
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )
        _freeze_fields(self, "result_snapshot", "target_configurations")


@dataclass(frozen=True, slots=True)
class AnalysisExportPreparationFailed(_SealedMessage):
    request_id: str
    analysis_id: str
    source_id: str
    record_id: str
    workflow_generation: int
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(
            self, "request_id", "analysis_id", "source_id", "record_id"
        )
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )
        _normalize_text_fields(self, "reason")


@dataclass(frozen=True, slots=True)
class ManualLabelExportPrepared(_SealedMessage):
    request_id: str
    command_id: str
    record_id: str
    label: str
    workflow_generation: int
    result_snapshot: Any
    target_configurations: tuple[Any, ...]

    def __post_init__(self) -> None:
        _validate_identifiers(self, "request_id", "command_id", "record_id")
        _normalize_text_fields(self, "label")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )
        _freeze_fields(self, "result_snapshot", "target_configurations")


@dataclass(frozen=True, slots=True)
class ManualLabelExportPreparationFailed(_SealedMessage):
    request_id: str
    command_id: str
    record_id: str
    label: str
    workflow_generation: int
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "request_id", "command_id", "record_id")
        _normalize_text_fields(self, "label", "reason")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )


@dataclass(frozen=True, slots=True)
class ExportPreparationCancelled(_SealedMessage):
    request_id: str
    workflow_generation: int

    def __post_init__(self) -> None:
        _validate_identifiers(self, "request_id")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )


@dataclass(frozen=True, slots=True)
class AnalysisTransportReady(_SealedMessage):
    analysis_id: str
    source_id: str
    record_id: str
    workflow_generation: int
    payload: Any

    def __post_init__(self) -> None:
        _validate_identifiers(self, "analysis_id", "source_id", "record_id")
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )
        _freeze_fields(self, "payload")


@dataclass(frozen=True, slots=True)
class AnalysisFailed(_SealedMessage):
    analysis_id: str
    source_id: str
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "analysis_id", "source_id")
        _normalize_text_fields(self, "reason")


@dataclass(frozen=True, slots=True)
class ExportCompleted(_SealedMessage):
    job_id: str
    attempt_id: str
    record_id: str
    target_results: tuple[Any, ...]

    def __post_init__(self) -> None:
        _validate_identifiers(self, "job_id", "attempt_id", "record_id")
        _freeze_fields(self, "target_results")


@dataclass(frozen=True, slots=True)
class ExportFailed(_SealedMessage):
    job_id: str
    attempt_id: str
    record_id: str
    failures: tuple[Any, ...]

    def __post_init__(self) -> None:
        _validate_identifiers(self, "job_id", "attempt_id", "record_id")
        _freeze_fields(self, "failures")


@dataclass(frozen=True, slots=True)
class ExportRetryAccepted(_SealedMessage):
    """Exact acknowledgement that a retry attempt is installed."""

    job_id: str
    previous_attempt_id: str
    new_attempt_id: str
    attempt_number: int

    def __post_init__(self) -> None:
        _validate_identifiers(
            self, "job_id", "previous_attempt_id", "new_attempt_id"
        )
        if type(self.attempt_number) is not int or self.attempt_number < 2:
            raise ValueError("attempt_number must be an integer greater than one")
        if self.previous_attempt_id == self.new_attempt_id:
            raise ValueError("retry acknowledgement requires a new attempt")


@dataclass(frozen=True, slots=True)
class RecordingLabelCommitted(_SealedMessage):
    command_id: str
    record_id: str
    label: str
    outcome: Any

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "record_id")
        _normalize_text_fields(self, "label")
        _freeze_fields(self, "outcome")


@dataclass(frozen=True, slots=True)
class RecordingLabelCommitFailed(_SealedMessage):
    command_id: str
    record_id: str
    label: str
    reason: str
    outcome: Any = None

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id", "record_id")
        _normalize_text_fields(self, "label", "reason")
        _freeze_fields(self, "outcome")


@dataclass(frozen=True, slots=True)
class WorkflowCommandRejected(_SealedMessage):
    command_id: str
    current_phase: Any
    reason: str

    def __post_init__(self) -> None:
        _validate_identifiers(self, "command_id")
        _normalize_text_fields(self, "reason")
        _freeze_fields(self, "current_phase")


@dataclass(frozen=True, slots=True)
class WorkflowStateChanged(_SealedMessage):
    workflow_generation: int
    previous_phase: Any
    new_phase: Any
    active_session_id: str | None = None
    active_import_id: str | None = None
    active_analysis_id: str | None = None
    active_job_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workflow_generation",
            _require_generation("workflow_generation", self.workflow_generation),
        )
        for field_name in (
            "active_session_id",
            "active_import_id",
            "active_analysis_id",
            "active_job_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _validate_identifiers(self, field_name)
        _freeze_fields(self, "previous_phase", "new_phase")


@dataclass(frozen=True, slots=True)
class ConfigurationChanged(_SealedMessage):
    configuration_generation: int
    configuration_snapshot: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration_generation",
            _require_generation("configuration_generation", self.configuration_generation),
        )
        _freeze_fields(self, "configuration_snapshot")


@dataclass(frozen=True, slots=True)
class ShutdownFlushFailed(_SealedMessage):
    shutdown_generation: int
    job_id: str
    attempt_id: str
    failures: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )
        _validate_identifiers(self, "job_id", "attempt_id")
        _freeze_fields(self, "failures")


@dataclass(frozen=True, slots=True)
class ShutdownFlushCompleted(_SealedMessage):
    shutdown_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )


@dataclass(frozen=True, slots=True)
class ShutdownAborted(_SealedMessage):
    shutdown_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )


@dataclass(frozen=True, slots=True)
class ShutdownReady(_SealedMessage):
    shutdown_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shutdown_generation",
            _require_generation("shutdown_generation", self.shutdown_generation),
        )
