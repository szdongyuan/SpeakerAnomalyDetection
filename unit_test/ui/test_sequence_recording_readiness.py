from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from ui.sequence.sequence_messages import ConfigurationSnapshot
from ui.sequence.sequence_recording_service import (
    RecordingReadinessRuntimeCapabilities,
    RecordingReadinessSnapshot,
    SequenceRecordingReadinessService,
)
from ui.sequence.sequence_recording_view import SequenceRecordingView


ROOT = Path(__file__).resolve().parents[2]
WIDGET = ROOT / "ui" / "sequence" / "sequence_widget.py"
SERVICE = ROOT / "ui" / "sequence" / "sequence_recording_service.py"

MISSING_CONFIGURATION = (
    "未找到可用配置。\n"
    "请先在上方【使用配置】下拉框中选择配置；\n"
    "如无可选项，请到【功能-测试队列】中保存或导入配置。"
)
MISSING_SPEAKER = "未选择扬声器，请在【硬件-硬件选择】中选择扬声器。"
DEFAULT_DEVICE_FAILURE = (
    "音频设备不可用，请检查设备连接或在【硬件-硬件选择】中重新选择设备。"
)


def _configuration(
    mode="RECORD_ONLY",
    detail=None,
    *,
    mic=None,
    speaker=None,
):
    sequence = [] if mode is None else [
        {"seq1": {"acq": {"mode": mode, "detail": dict(detail or {})}}}
    ]
    return ConfigurationSnapshot(
        sequence_config=sequence,
        analysis_config={},
        mic={"name": "mic", "samplerate": 48_000} if mic is None else mic,
        speaker=(
            {"name": "speaker", "samplerate": 48_000}
            if speaker is None
            else speaker
        ),
    )


def _command(command_id="start-1", generation=7):
    return SimpleNamespace(
        command_id=command_id,
        configuration_generation=generation,
    )


def _runtime(available=True, message=""):
    return RecordingReadinessRuntimeCapabilities(
        audio_devices_available=available,
        audio_devices_unavailable_message=message,
    )


def test_readiness_freezes_exact_recording_inputs_and_uses_input_runtime_rate():
    calls = []
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda mic: calls.append(mic)
        or SimpleNamespace(ok=True, sample_rate=48_000, message=""),
        duplex_sample_rate_resolver=lambda *_args: pytest.fail(
            "record-only without monitoring must not query duplex rate"
        ),
    )
    configuration = _configuration(
        detail={"sample_rate": 44_100, "monitor_playback": "False"},
        speaker=False,
    )

    result = service.assess(_command(), configuration)

    assert result.ready is True
    assert result.reason == ""
    assert result.runtime_sample_rate == 48_000
    assert type(result.snapshot) is RecordingReadinessSnapshot
    assert result.snapshot.configuration_generation == 7
    assert result.snapshot.recording_mode == "RECORD_ONLY"
    assert result.snapshot.monitor_playback is False
    assert result.snapshot.speaker_required is False
    assert result.snapshot.audio_devices_unavailable_message == ""
    assert result.snapshot.input_sample_rate_source["samplerate"] == 48_000
    assert result.snapshot.output_sample_rate_source is None
    assert type(result.snapshot.sequence_acquisition_config) is MappingProxyType
    with pytest.raises(TypeError):
        result.snapshot.sequence_acquisition_config["mode"] = "changed"
    assert len(calls) == 1
    calls[0]["samplerate"] = 44_100
    assert result.snapshot.input_sample_rate_source["samplerate"] == 48_000


@pytest.mark.parametrize("mode", ["PLAY_AND_RECORD", "RECORD_ONLY"])
def test_speaker_requirement_rejects_before_sample_rate_query_with_exact_warning(mode):
    warnings = []
    view = SequenceRecordingView(
        present_readiness_warning=lambda title, text: warnings.append((title, text))
    )
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        view=view,
        input_sample_rate_resolver=lambda *_args: pytest.fail("must not resolve input"),
        duplex_sample_rate_resolver=lambda *_args: pytest.fail("must not resolve duplex"),
    )
    detail = {"monitor_playback": True} if mode == "RECORD_ONLY" else {}
    configuration = _configuration(mode, detail, speaker=False)

    assert service(_command(), configuration) == (
        False,
        "recording preflight rejected",
    )
    assert warnings == [("提示", MISSING_SPEAKER)]


def test_readiness_preserves_duplex_mismatch_warning_and_runtime_authority():
    mismatch = "输入设备与输出设备的采样率不一致，请在硬件管理中设置为一致后重新选择。"
    warnings = []
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        view=SequenceRecordingView(
            present_readiness_warning=lambda title, text: warnings.append((title, text))
        ),
    )
    configuration = _configuration(
        "PLAY_AND_RECORD",
        {"sample_rate": 32_000},
        mic={"name": "mic", "samplerate": 44_100},
        speaker={"name": "speaker", "samplerate": 48_000},
    )

    result = service.assess(_command(), configuration)

    assert result.ready is False
    assert result.runtime_sample_rate is None
    assert result.warning_text == mismatch
    assert warnings == [("提示", mismatch)]


@pytest.mark.parametrize(
    "invalid_ok",
    [
        1,
        0,
        None,
        "yes",
        "",
        [],
        {},
        object(),
        pytest.param(np.bool_(True), id="numpy-bool"),
    ],
)
def test_resolver_ok_requires_an_exact_boolean_without_truth_coercion(invalid_ok):
    class Resolution:
        ok = invalid_ok
        sample_rate = 48_000
        message = ""

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: Resolution(),
    )

    result = service.assess(_command(), _configuration())

    assert result.ready is False
    assert result.reason == "recording readiness snapshot is unavailable"
    assert result.runtime_sample_rate is None
    assert result.warning_text == "录音启动检查失败，请重试。"
    assert type(result.snapshot) is RecordingReadinessSnapshot
    with pytest.raises((AttributeError, TypeError)):
        result.snapshot.recording_mode = "changed"


@pytest.mark.parametrize(
    ("sample_rate", "expected"),
    [
        (44_100, 44_100),
        (48_000.0, 48_000),
        (np.int64(48_000), 48_000),
        (np.float64(44_100.0), 44_100),
    ],
)
def test_valid_input_resolver_rate_is_a_detached_python_integer(sample_rate, expected):
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: SimpleNamespace(
            ok=True,
            sample_rate=sample_rate,
            message="ignored-on-success",
        ),
    )

    result = service.assess(_command(), _configuration())

    assert result.ready is True
    assert result.runtime_sample_rate == expected
    assert type(result.runtime_sample_rate) is int


def test_valid_duplex_resolver_rate_uses_same_strict_scalar_contract():
    calls = []
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda *_args: pytest.fail("must use duplex"),
        duplex_sample_rate_resolver=lambda mic, speaker: calls.append((mic, speaker))
        or SimpleNamespace(
            ok=True,
            sample_rate=np.float64(48_000.0),
            message="",
        ),
    )

    result = service.assess(
        _command(),
        _configuration("RECORD_ONLY", {"monitor_playback": True}),
    )

    assert result.ready is True
    assert result.runtime_sample_rate == 48_000
    assert type(result.runtime_sample_rate) is int
    assert len(calls) == 1


class _HostileNumeric:
    def __init__(self):
        self.calls = []

    def __bool__(self):
        self.calls.append("bool")
        raise KeyboardInterrupt("must not coerce bool")

    def __float__(self):
        self.calls.append("float")
        raise SystemExit("must not coerce float")

    def __int__(self):
        self.calls.append("int")
        raise RuntimeError("must not coerce int")

    def __str__(self):
        self.calls.append("str")
        raise RuntimeError("must not coerce str")


class _HostileFloat(float):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.calls = []
        return instance

    def is_integer(self):
        self.calls.append("is_integer")
        raise KeyboardInterrupt("must not inspect hostile float subclass")

    def __float__(self):
        self.calls.append("float")
        raise SystemExit("must not coerce hostile float subclass")


class _HostileStr(str):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.calls = []
        return instance

    def strip(self, *_args, **_kwargs):
        self.calls.append("strip")
        raise KeyboardInterrupt("must not strip hostile str subclass")

    def __getitem__(self, _item):
        self.calls.append("getitem")
        raise SystemExit("must not slice hostile str subclass")


@pytest.mark.parametrize(
    "invalid_rate",
    [
        True,
        False,
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        -48_000,
        -1,
        0,
        48_000.5,
        32_000,
        "48000",
        [48_000],
        {"sample_rate": 48_000},
        pytest.param(np.array(48_000), id="numpy-scalar-array"),
        pytest.param(np.array([48_000]), id="numpy-array"),
    ],
)
def test_success_resolver_rejects_unsupported_or_non_scalar_rates(invalid_rate):
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: SimpleNamespace(
            ok=True,
            sample_rate=invalid_rate,
            message="",
        ),
    )

    result = service.assess(_command(), _configuration())

    assert result.ready is False
    assert result.reason == "recording readiness snapshot is unavailable"
    assert result.runtime_sample_rate is None


def test_success_resolver_never_invokes_hostile_numeric_conversion_methods():
    hostile = _HostileNumeric()
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: SimpleNamespace(
            ok=True,
            sample_rate=hostile,
            message="",
        ),
    )

    assert service(_command(), _configuration())[0] is False
    assert hostile.calls == []


def test_success_resolver_rejects_numeric_subclass_without_invoking_methods():
    hostile = _HostileFloat(48_000.0)
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: SimpleNamespace(
            ok=True,
            sample_rate=hostile,
            message="",
        ),
    )

    assert service(_command(), _configuration())[0] is False
    assert hostile.calls == []


def test_failed_resolver_ignores_rate_entirely_and_bounds_plain_warning_text():
    class FailedResolution:
        ok = False
        message = "x" * 10_000

        @property
        def sample_rate(self):
            raise AssertionError("failed resolution rate must never be read")

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: FailedResolution(),
    )

    result = service.assess(_command(), _configuration())

    assert result.ready is False
    assert result.runtime_sample_rate is None
    assert type(result.warning_text) is str
    assert result.warning_text == "x" * 2_048


def test_failed_resolver_rejects_hostile_non_plain_warning_without_conversion():
    hostile = _HostileNumeric()
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: SimpleNamespace(
            ok=False,
            sample_rate=hostile,
            message=hostile,
        ),
    )

    result = service.assess(_command(), _configuration())

    assert result.ready is False
    assert result.reason == "recording preflight rejected"
    assert result.warning_text == "录音启动检查失败，请重试。"
    assert hostile.calls == []


@pytest.mark.parametrize("message", ["", "   ", "\t\r\n"])
def test_failed_resolver_empty_or_whitespace_warning_uses_stable_generic_text(message):
    warnings = []
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: SimpleNamespace(
            ok=False,
            sample_rate=object(),
            message=message,
        ),
        view=SequenceRecordingView(
            present_readiness_warning=lambda title, text: warnings.append((title, text))
        ),
    )

    result = service.assess(_command(), _configuration())

    assert result.ready is False
    assert result.warning_text == "录音启动检查失败，请重试。"
    assert warnings == [("提示", "录音启动检查失败，请重试。")]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("x" * 2_048, "x" * 2_048),
        ("x" * 2_049, "x" * 2_048),
        (" " * 2_047 + "x" + "tail", " " * 2_047 + "x"),
        (" " * 2_048 + "x", "录音启动检查失败，请重试。"),
        (" " * 10_000 + "tail", "录音启动检查失败，请重试。"),
        ("\u2003" * 2_048 + "x", "录音启动检查失败，请重试。"),
        ("\u2003\u2002\u3000", "录音启动检查失败，请重试。"),
    ],
)
def test_resolver_warning_is_nonblank_after_bounded_truncation(message, expected):
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: SimpleNamespace(
            ok=False,
            sample_rate=None,
            message=message,
        ),
    )

    result = service.assess(_command(), _configuration())

    assert type(result.warning_text) is str
    assert len(result.warning_text) <= 2_048
    assert result.warning_text.strip()
    assert result.warning_text == expected


def test_hostile_str_subclass_warning_uses_fallback_without_methods():
    hostile = _HostileStr("device down")
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: SimpleNamespace(
            ok=False,
            sample_rate=None,
            message=hostile,
        ),
    )

    result = service.assess(_command(), _configuration())

    assert result.warning_text == "录音启动检查失败，请重试。"
    assert hostile.calls == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("   ", DEFAULT_DEVICE_FAILURE),
        ("x" * 10_000, "x" * 2_048),
    ],
)
def test_device_warning_is_nonblank_plain_and_bounded(message, expected):
    runtime = _runtime(False, message)
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: runtime,
    )

    result = service.assess(_command(), _configuration())

    assert result.ready is False
    assert type(result.warning_text) is str
    assert result.warning_text == expected
    assert result.snapshot.audio_devices_unavailable_message == expected


def test_hostile_device_warning_uses_fallback_without_conversion():
    hostile = _HostileNumeric()
    runtime = _runtime(False, "safe")
    object.__setattr__(runtime, "audio_devices_unavailable_message", hostile)
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: runtime,
    )

    result = service.assess(_command(), _configuration())

    assert result.ready is False
    assert result.warning_text == DEFAULT_DEVICE_FAILURE
    assert result.snapshot.audio_devices_unavailable_message == DEFAULT_DEVICE_FAILURE
    assert hostile.calls == []


def test_cached_result_is_detached_from_mutated_resolver_and_configuration_inputs():
    resolution = SimpleNamespace(ok=True, sample_rate=48_000, message="")
    sequence = [
        {
            "seq1": {
                "acq": {
                    "mode": "RECORD_ONLY",
                    "detail": {"monitor_playback": False},
                }
            }
        }
    ]
    mic = {"name": "mic", "samplerate": 48_000, "nested": [1]}
    configuration = ConfigurationSnapshot(
        sequence_config=sequence,
        analysis_config={},
        mic=mic,
        speaker=False,
    )
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: resolution,
    )

    first = service.assess(_command(), configuration)
    resolution.ok = False
    resolution.sample_rate = 44_100
    resolution.message = "changed"
    sequence[0]["seq1"]["acq"]["mode"] = "PLAY_AND_RECORD"
    mic["samplerate"] = 44_100
    mic["nested"].append(2)
    second = service.assess(_command(), configuration)

    assert second is first
    assert first.ready is True
    assert first.runtime_sample_rate == 48_000
    assert type(first.runtime_sample_rate) is int
    assert first.snapshot.recording_mode == "RECORD_ONLY"
    assert first.snapshot.input_sample_rate_source["samplerate"] == 48_000
    assert first.snapshot.input_sample_rate_source["nested"] == (1,)
    with pytest.raises((AttributeError, TypeError)):
        first.runtime_sample_rate = 44_100


@pytest.mark.parametrize(
    ("configuration", "runtime", "expected"),
    [
        (_configuration(mode=None), _runtime(), MISSING_CONFIGURATION),
        (_configuration(), _runtime(False, "设备不可用"), "设备不可用"),
        (_configuration(), _runtime(False, ""), DEFAULT_DEVICE_FAILURE),
    ],
)
def test_configuration_and_device_failures_keep_exact_warning_contract(
    configuration, runtime, expected
):
    warnings = []
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: runtime,
        view=SequenceRecordingView(
            present_readiness_warning=lambda title, text: warnings.append((title, text))
        ),
    )

    assert service(_command(), configuration)[0] is False
    assert warnings == [("提示", expected)]


@pytest.mark.parametrize(
    ("mic", "speaker", "expected"),
    [
        (False, {"name": "speaker"}, "未找到麦克风，请在硬件中设置"),
        ({"name": "mic"}, False, "未找到扬声器，请在硬件中设置"),
    ],
)
def test_malformed_nonempty_acquisition_keeps_legacy_device_fallback(
    mic, speaker, expected
):
    warnings = []
    configuration = ConfigurationSnapshot(
        sequence_config=[{"seq1": {}}],
        analysis_config={},
        mic=mic,
        speaker=speaker,
    )
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        view=SequenceRecordingView(
            present_readiness_warning=lambda title, text: warnings.append((title, text))
        ),
    )

    assert service(_command(), configuration)[0] is False
    assert warnings == [("提示", expected)]


def test_unknown_acquisition_mode_with_both_devices_keeps_legacy_acceptance():
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda *_args: pytest.fail("must not resolve input"),
        duplex_sample_rate_resolver=lambda *_args: pytest.fail("must not resolve duplex"),
    )

    assert service(_command(), _configuration("UNKNOWN")) == (True, "")


@pytest.mark.parametrize("failure", [RuntimeError("ordinary"), KeyboardInterrupt(), SystemExit()])
def test_hostile_runtime_provider_fails_closed_once_and_next_command_retries(failure):
    calls = []

    def provider():
        calls.append("provider")
        if len(calls) == 1:
            raise failure
        return _runtime()

    warnings = []
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=provider,
        view=SequenceRecordingView(
            present_readiness_warning=lambda title, text: warnings.append((title, text))
        ),
    )
    configuration = _configuration()

    first = service(_command("first"), configuration)
    duplicate = service(_command("first"), configuration)
    retry = service(_command("retry"), configuration)

    assert first == duplicate == (False, "recording readiness snapshot is unavailable")
    assert retry == (True, "")
    assert calls == ["provider", "provider"]
    assert warnings == [("提示", "录音启动检查失败，请重试。")]


@pytest.mark.parametrize("failure", [RuntimeError("ordinary"), KeyboardInterrupt(), SystemExit()])
def test_hostile_sample_rate_provider_fails_closed_without_duplicate_side_effects(
    failure,
):
    resolver_calls = []
    warnings = []

    def resolver(_mic):
        resolver_calls.append("resolver")
        raise failure

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=resolver,
        view=SequenceRecordingView(
            present_readiness_warning=lambda title, text: warnings.append((title, text))
        ),
    )
    configuration = _configuration()

    assert service(_command(), configuration) == (
        False,
        "recording readiness snapshot is unavailable",
    )
    assert service(_command(), configuration) == (
        False,
        "recording readiness snapshot is unavailable",
    )
    assert resolver_calls == ["resolver"]
    assert warnings == [("提示", "录音启动检查失败，请重试。")]


@pytest.mark.parametrize("failure", [RuntimeError("ordinary"), KeyboardInterrupt(), SystemExit()])
def test_hostile_warning_presentation_is_terminally_deduplicated_and_retryable(failure):
    attempts = []

    def present(_title, _text):
        attempts.append("present")
        raise failure

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(False, "device down"),
        view=SequenceRecordingView(present_readiness_warning=present),
    )
    configuration = _configuration()

    assert service(_command("one"), configuration)[0] is False
    assert service(_command("one"), configuration)[0] is False
    assert service(_command("two"), configuration)[0] is False
    assert attempts == ["present", "present", "present"]


def test_false_presentation_retries_cached_result_then_success_is_exactly_once():
    provider_calls = []
    outcomes = iter([False, None])
    attempts = []

    def present(title, text):
        attempts.append((title, text))
        return next(outcomes)

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: provider_calls.append("provider")
        or _runtime(False, "device down"),
        view=SequenceRecordingView(present_readiness_warning=present),
    )
    command = _command("presentation-retry")
    configuration = _configuration()

    first = service.assess(command, configuration)
    second = service.assess(command, configuration)
    third = service.assess(command, configuration)

    assert first is second is third
    assert provider_calls == ["provider"]
    assert attempts == [("提示", "device down"), ("提示", "device down")]


def test_concurrent_duplicate_returns_immediate_nack_then_observes_cached_terminal():
    provider_calls = []
    warning_calls = []
    configuration = _configuration()
    command = _command("same")
    provider_barrier = Barrier(2)
    release_provider = Event()

    def provider():
        provider_calls.append("provider")
        provider_barrier.wait(timeout=5)
        assert release_provider.wait(5)
        return _runtime(False, "device down")

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=provider,
        view=SequenceRecordingView(
            present_readiness_warning=lambda title, text: warning_calls.append(
                (title, text)
            )
        ),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(service.assess, command, configuration)
        provider_barrier.wait(timeout=5)
        duplicate = executor.submit(service.assess, command, configuration)
        try:
            nack = duplicate.result(timeout=1)
        finally:
            release_provider.set()
        terminal = owner.result(timeout=5)

    assert nack.reason == "recording readiness re-entry rejected"
    assert nack.warning_text == "录音启动检查正在进行，请重试。"
    assert terminal.reason == "recording preflight rejected"
    assert service.assess(command, configuration) is terminal
    assert provider_calls == ["provider"]
    assert warning_calls == [("提示", "device down")]


def test_reentrant_snapshot_provider_fails_closed_once_and_next_request_recovers():
    provider_calls = []
    nested_results = []
    warnings = []
    configuration = _configuration()
    command = _command("reentrant")
    service = None

    def provider():
        provider_calls.append("provider")
        if len(provider_calls) == 1:
            nested_results.append(service(command, configuration))
        return _runtime()

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=provider,
        view=SequenceRecordingView(
            present_readiness_warning=lambda title, text: warnings.append((title, text))
        ),
    )

    assert service(command, configuration) == (True, "")
    assert nested_results == [(False, "recording readiness re-entry rejected")]
    assert provider_calls == ["provider"]
    assert warnings == []
    assert service.assess(command, configuration).ready is True
    assert service(_command("next"), configuration) == (True, "")
    assert provider_calls == ["provider", "provider"]


def test_provider_can_join_cross_thread_nested_assess_without_lock_deadlock():
    nested = []
    blocked = []
    threads = []
    command = _command("provider-cross-thread")
    configuration = _configuration()
    service = None

    def provider():
        thread = Thread(target=lambda: nested.append(service.assess(command, configuration)))
        threads.append(thread)
        thread.start()
        thread.join(1)
        blocked.append(thread.is_alive())
        return _runtime()

    service = SequenceRecordingReadinessService(runtime_capabilities_provider=provider)

    terminal = service.assess(command, configuration)
    for thread in threads:
        thread.join(5)

    assert blocked == [False]
    assert [item.reason for item in nested] == ["recording readiness re-entry rejected"]
    assert terminal.ready is True
    assert service.assess(command, configuration) is terminal


def test_resolver_can_join_cross_thread_nested_assess_without_lock_deadlock():
    nested = []
    blocked = []
    threads = []
    resolver_calls = []
    command = _command("resolver-cross-thread")
    configuration = _configuration()
    service = None

    def resolver(_mic):
        resolver_calls.append("resolver")
        thread = Thread(target=lambda: nested.append(service.assess(command, configuration)))
        threads.append(thread)
        thread.start()
        thread.join(1)
        blocked.append(thread.is_alive())
        return SimpleNamespace(ok=True, sample_rate=48_000, message="")

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=resolver,
    )

    terminal = service.assess(command, configuration)
    for thread in threads:
        thread.join(5)

    assert blocked == [False]
    assert [item.reason for item in nested] == ["recording readiness re-entry rejected"]
    assert terminal.ready is True
    assert resolver_calls == ["resolver"]
    assert service.assess(command, configuration) is terminal


def test_view_can_join_cross_thread_cached_assess_without_lock_deadlock_or_duplicate():
    nested = []
    blocked = []
    threads = []
    presentation_calls = []
    command = _command("view-cross-thread")
    configuration = _configuration()
    service = None

    def present(title, text):
        presentation_calls.append((title, text))
        thread = Thread(target=lambda: nested.append(service.assess(command, configuration)))
        threads.append(thread)
        thread.start()
        thread.join(1)
        blocked.append(thread.is_alive())

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(False, "device down"),
        view=SequenceRecordingView(present_readiness_warning=present),
    )

    terminal = service.assess(command, configuration)
    for thread in threads:
        thread.join(5)

    assert blocked == [False]
    assert nested == [terminal]
    assert presentation_calls == [("提示", "device down")]
    assert service.assess(command, configuration) is terminal


def test_logger_can_join_cross_thread_nested_assess_without_lock_deadlock():
    nested = []
    blocked = []
    threads = []
    log_calls = []
    command = _command("logger-cross-thread")
    configuration = _configuration()
    service = None

    def warning(message):
        log_calls.append(message)
        thread = Thread(target=lambda: nested.append(service.assess(command, configuration)))
        threads.append(thread)
        thread.start()
        thread.join(1)
        blocked.append(thread.is_alive())

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: (_ for _ in ()).throw(
            RuntimeError("resolver failed")
        ),
        logger=SimpleNamespace(warning=warning),
    )

    terminal = service.assess(command, configuration)
    for thread in threads:
        thread.join(5)

    assert blocked == [False]
    assert [item.reason for item in nested] == ["recording readiness re-entry rejected"]
    assert terminal.reason == "recording readiness snapshot is unavailable"
    assert len(log_calls) == 1
    assert service.assess(command, configuration) is terminal


class _HostileFormattedError(RuntimeError):
    def __init__(self, failure_type):
        super().__init__("hostile formatting")
        self.failure_type = failure_type

    def __str__(self):
        raise self.failure_type("error formatting failed")


@pytest.mark.parametrize(
    "failure_type",
    [RuntimeError, KeyboardInterrupt, SystemExit],
)
@pytest.mark.parametrize("boundary", ["getter", "call", "format"])
def test_logger_boundary_never_leaks_or_leaves_identity_inflight(
    failure_type,
    boundary,
):
    resolver_calls = []

    if boundary == "getter":
        class Logger:
            @property
            def warning(self):
                raise failure_type("logger getter failed")

        logger = Logger()
        resolver_error = RuntimeError("resolver failed")
    elif boundary == "call":
        def warning(_message):
            raise failure_type("logger call failed")

        logger = SimpleNamespace(warning=warning)
        resolver_error = RuntimeError("resolver failed")
    else:
        logger = SimpleNamespace(
            warning=lambda _message: pytest.fail("format failure must precede logger call")
        )
        resolver_error = _HostileFormattedError(failure_type)

    def resolver(_mic):
        resolver_calls.append("resolver")
        raise resolver_error

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=resolver,
        logger=logger,
    )
    command = _command(f"logger-{boundary}-{failure_type.__name__}")
    configuration = _configuration()

    first = service.assess(command, configuration)
    second = service.assess(command, configuration)

    assert first is second
    assert first.reason == "recording readiness snapshot is unavailable"
    assert resolver_calls == ["resolver"]
    assert service.completed_request_count == 1


@pytest.mark.parametrize("failure", [RuntimeError("ordinary"), KeyboardInterrupt(), SystemExit()])
def test_unexpected_internal_logger_failure_still_commits_fallback_terminal(failure):
    resolver_calls = []
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(),
        input_sample_rate_resolver=lambda _mic: resolver_calls.append("resolver")
        or (_ for _ in ()).throw(RuntimeError("resolver failed")),
    )
    service._log_failure = lambda *_args: (_ for _ in ()).throw(failure)
    command = _command(f"internal-logger-{type(failure).__name__}")
    configuration = _configuration()

    first = service.assess(command, configuration)
    second = service.assess(command, configuration)

    assert first is second
    assert first.reason == "recording readiness snapshot is unavailable"
    assert resolver_calls == ["resolver"]


@pytest.mark.parametrize("failure", [RuntimeError("ordinary"), KeyboardInterrupt(), SystemExit()])
def test_unexpected_internal_presentation_failure_returns_cached_terminal(failure):
    provider_calls = []
    presentation_calls = []
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: provider_calls.append("provider")
        or _runtime(False, "device down"),
        view=SimpleNamespace(present_readiness_warning=lambda *_args: None),
    )

    def present(*_args):
        presentation_calls.append("present")
        raise failure

    service._present_reserved = present
    command = _command(f"internal-presentation-{type(failure).__name__}")
    configuration = _configuration()

    first = service.assess(command, configuration)
    second = service.assess(command, configuration)

    assert first is second
    assert first.reason == "recording preflight rejected"
    assert provider_calls == ["provider"]
    assert presentation_calls == ["present", "present"]


def test_presentation_failure_logger_nested_assess_cannot_duplicate_presentation():
    nested = []
    blocked = []
    presentation_calls = []
    log_calls = []
    threads = []
    command = _command("presentation-logger-cross-thread")
    configuration = _configuration()
    service = None

    def present(_title, _text):
        presentation_calls.append("present")
        raise RuntimeError("presentation failed")

    def warning(message):
        log_calls.append(message)
        if len(log_calls) == 1:
            thread = Thread(
                target=lambda: nested.append(service.assess(command, configuration))
            )
            threads.append(thread)
            thread.start()
            thread.join(1)
            blocked.append(thread.is_alive())

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: _runtime(False, "device down"),
        view=SimpleNamespace(present_readiness_warning=present),
        logger=SimpleNamespace(warning=warning),
    )

    terminal = service.assess(command, configuration)
    for thread in threads:
        thread.join(5)

    assert blocked == [False]
    assert nested == [terminal]
    assert presentation_calls == ["present"]
    assert len(log_calls) == 1


def test_commit_protects_new_terminal_while_older_presentation_is_blocked():
    provider_calls = []
    presentation_entered = Event()
    release_presentation = Event()
    nested = []
    older_command = _command("older-blocked")
    newer_command = _command("newer-terminal")
    configuration = _configuration()
    service = None

    def provider():
        provider_calls.append("provider")
        return _runtime(False, "device down") if len(provider_calls) == 1 else _runtime()

    def present(_title, _text):
        nested.append(service.assess(older_command, configuration))
        presentation_entered.set()
        assert release_presentation.wait(5)

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=provider,
        view=SequenceRecordingView(present_readiness_warning=present),
        completed_request_limit=1,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        older_future = executor.submit(service.assess, older_command, configuration)
        assert presentation_entered.wait(5)
        newer = service.assess(newer_command, configuration)
        duplicate_newer = service.assess(newer_command, configuration)
        assert duplicate_newer is newer
        assert provider_calls == ["provider", "provider"]
        assert service.completed_request_count == 2
        release_presentation.set()
        older = older_future.result(timeout=5)

    assert nested == [older]
    assert service.completed_request_count == 1
    assert list(service._completed) == [("newer-terminal", 7)]
    assert service.assess(newer_command, configuration) is newer
    assert provider_calls == ["provider", "provider"]


def test_presentation_and_completed_capacity_backpressures_without_provider_call():
    provider_calls = []
    presentation_calls = []
    all_presentations_entered = Event()
    release_presentations = Event()
    presentation_lock = Lock()
    configuration = _configuration()

    def provider():
        provider_calls.append("provider")
        return _runtime(False, "device down")

    def present(_title, _text):
        with presentation_lock:
            presentation_calls.append("present")
            if len(presentation_calls) == 2:
                all_presentations_entered.set()
        assert release_presentations.wait(5)

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=provider,
        view=SequenceRecordingView(present_readiness_warning=present),
        completed_request_limit=1,
        presenting_limit=2,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        blocked = [
            executor.submit(service.assess, _command(f"blocked-{index}"), configuration)
            for index in range(2)
        ]
        assert all_presentations_entered.wait(5)
        deferred_presentation = service.assess(_command("deferred"), configuration)
        assert deferred_presentation.ready is False
        assert service.assess(_command("deferred"), configuration) is deferred_presentation
        overflow = service.assess(_command("overflow"), configuration)
        assert overflow.reason == "recording readiness capacity unavailable"
        assert provider_calls == ["provider", "provider", "provider"]
        assert service.completed_request_count == 3
        assert len(service._presenting) == 2
        release_presentations.set()
        [future.result(timeout=5) for future in blocked]

    assert service.completed_request_count == 1
    retry = service.assess(_command("overflow"), configuration)
    assert retry.reason == "recording preflight rejected"
    assert provider_calls == ["provider", "provider", "provider", "provider"]


@pytest.mark.parametrize(
    "outcome",
    [False, KeyboardInterrupt(), SystemExit()],
)
def test_failed_blocked_presentation_releases_capacity_and_trims_safely(outcome):
    provider_calls = []
    presentation_entered = Event()
    release_presentation = Event()
    configuration = _configuration()

    def provider():
        provider_calls.append("provider")
        return _runtime(False, "device down") if len(provider_calls) == 1 else _runtime()

    def present(_title, _text):
        presentation_entered.set()
        assert release_presentation.wait(5)
        if outcome is False:
            return False
        raise outcome

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=provider,
        view=SequenceRecordingView(present_readiness_warning=present),
        completed_request_limit=1,
        presenting_limit=1,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        blocked = executor.submit(service.assess, _command("blocked"), configuration)
        assert presentation_entered.wait(5)
        newer = service.assess(_command("newer"), configuration)
        assert service.assess(_command("newer"), configuration) is newer
        overflow = service.assess(_command("overflow"), configuration)
        assert overflow.reason == "recording readiness capacity unavailable"
        assert provider_calls == ["provider", "provider"]
        release_presentation.set()
        blocked.result(timeout=5)

    assert service.completed_request_count == 1
    assert list(service._completed) == [("newer", 7)]
    retry = service.assess(_command("overflow"), configuration)
    assert retry.ready is True
    assert provider_calls == ["provider", "provider", "provider"]


def test_readiness_history_is_bounded_without_duplicate_side_effects():
    lock = Lock()
    calls = 0

    def provider():
        nonlocal calls
        with lock:
            calls += 1
        return _runtime()

    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=provider,
        completed_request_limit=3,
    )
    configuration = _configuration()
    for index in range(5):
        assert service(_command(f"command-{index}"), configuration) == (True, "")
    assert service.completed_request_count == 3
    assert calls == 5


def test_facade_contains_only_readiness_composition_and_no_owner_algorithm_or_imports():
    source = WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    window = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    method_names = {
        node.name for node in window.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    module_function_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "checked_work_status_message" not in method_names
    assert "_workflow_recording_readiness" not in method_names
    assert "_resolve_runtime_sample_rate_for_mode" not in module_function_names
    assert {
        "normalize_record_only_detail",
        "resolve_input_sample_rate",
        "resolve_duplex_sample_rate",
    }.isdisjoint(imported_names)
    assert "SequenceRecordingReadinessService" in imported_names
    assert "recording_readiness_service" in source


def test_recording_readiness_owner_does_not_import_or_call_controllers():
    source = SERVICE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    readiness = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceRecordingReadinessService"
    )
    imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    accessed = {
        node.attr for node in ast.walk(readiness) if isinstance(node, ast.Attribute)
    }

    assert not any(module.endswith("_controller") for module in imports)
    assert not any(name.endswith("_controller") for name in accessed)
