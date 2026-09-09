from types import SimpleNamespace

import pytest


def _snapshot(*analysis_types):
    display = [f"item-{index}" for index, _value in enumerate(analysis_types)]
    config = {"display_sequence": display}
    config.update({key: {"type": value} for key, value in zip(display, analysis_types)})
    return {"analysis_config": config}


@pytest.mark.parametrize("analysis_types", [
    (), ("SPL",), ("FBA",), ("SPEC",),
    ("SPL", "FBA"), ("SPEC", "SPL", "FBA"),
])
def test_fast_overlap_policy_accepts_only_approved_subsets_and_empty(analysis_types):
    from ui.sequence.recording_analysis_eligibility import (
        enabled_analysis_identifiers,
        fast_recording_overlap_eligible,
    )

    identifiers = enabled_analysis_identifiers(_snapshot(*analysis_types))

    assert fast_recording_overlap_eligible(identifiers, ())


@pytest.mark.parametrize("analysis_type", ["SPL", "FBA", "SPEC"])
def test_publication_only_excel_does_not_disable_approved_overlap(analysis_type):
    from ui.sequence.recording_analysis_eligibility import (
        enabled_analysis_identifiers,
        fast_recording_overlap_eligible,
    )

    identifiers = enabled_analysis_identifiers(_snapshot(analysis_type, "Excel"))

    assert identifiers == (analysis_type,)
    assert fast_recording_overlap_eligible(identifiers, ())


def test_excel_only_has_the_same_eligibility_as_empty_analysis():
    from ui.sequence.recording_analysis_eligibility import (
        enabled_analysis_identifiers,
        fast_recording_overlap_eligible,
    )

    identifiers = enabled_analysis_identifiers(_snapshot("Excel"))

    assert identifiers == ()
    assert fast_recording_overlap_eligible(identifiers, ())


def test_malformed_excel_named_entry_remains_ineligible():
    from ui.sequence.recording_analysis_eligibility import (
        enabled_analysis_identifiers,
        fast_recording_overlap_eligible,
    )

    identifiers = enabled_analysis_identifiers({
        "analysis_config": {
            "display_sequence": ["excel"],
            "excel": {},
        },
    })

    assert identifiers == ("EXCEL",)
    assert not fast_recording_overlap_eligible(identifiers, ())


@pytest.mark.parametrize("analysis_type", [
    "ED", "PD", "PM", "FFT", "CQT", "LOUD", "future-analysis",
])
@pytest.mark.parametrize("location", ["new", "unfinished"])
def test_fast_overlap_policy_rejects_other_or_unknown_identifier_anywhere(
    analysis_type, location,
):
    from ui.sequence.recording_analysis_eligibility import (
        enabled_analysis_identifiers,
        fast_recording_overlap_eligible,
    )

    approved = enabled_analysis_identifiers(_snapshot("SPL"))
    other = enabled_analysis_identifiers(_snapshot(analysis_type))
    new_identifiers = other if location == "new" else approved
    unfinished = (other,) if location == "unfinished" else (approved,)

    assert not fast_recording_overlap_eligible(new_identifiers, unfinished)


def test_enabled_identifiers_are_frozen_before_mutable_config_changes():
    from ui.sequence.recording_analysis_eligibility import (
        enabled_analysis_identifiers,
        fast_recording_overlap_eligible,
    )

    snapshot = _snapshot("SPL")
    identifiers = enabled_analysis_identifiers(snapshot)
    snapshot["analysis_config"]["item-0"]["type"] = "ED"

    assert identifiers == ("SPL",)
    assert fast_recording_overlap_eligible(identifiers, ())


def test_missing_enabled_item_is_an_unknown_identifier_and_is_ineligible():
    from ui.sequence.recording_analysis_eligibility import (
        enabled_analysis_identifiers,
        fast_recording_overlap_eligible,
    )

    identifiers = enabled_analysis_identifiers({
        "analysis_config": {"display_sequence": ["missing"]},
    })

    assert identifiers == ("MISSING",)
    assert not fast_recording_overlap_eligible(identifiers, ())


def test_unfinished_context_identifiers_come_from_its_frozen_snapshot():
    from ui.sequence.sequence_widget_recording_process_ops import (
        SequenceWidgetRecordingProcessOpsMixin,
    )

    frozen_old = _snapshot("ED")
    context = SimpleNamespace(
        final=True, failed=False, cancelled=False,
        recent_session_config_snapshot=frozen_old,
        enabled_analysis_identifiers=("ED",),
    )
    host = SimpleNamespace(
        analysis_config=_snapshot("SPL")["analysis_config"],
        _recording_process_contexts={"A": context},
    )
    frozen_old["analysis_config"]["item-0"]["type"] = "SPL"

    eligible = SequenceWidgetRecordingProcessOpsMixin._recording_analysis_overlap_eligible(
        host, _snapshot("SPL"))

    assert not eligible


@pytest.mark.parametrize("old_types, expected", [
    ((), True), (("SPL",), True), (("FBA", "SPEC"), True),
    (("ED",), False), (("PD",), False), (("PM",), False),
    (("FFT",), False), (("CQT",), False), (("future-analysis",), False),
])
def test_calibration_uses_same_policy_for_unfinished_main_contexts(old_types, expected):
    from ui.calibration_window import InputCalibration

    service = SimpleNamespace(can_start_recording=True, busy=True)
    calibration = SimpleNamespace(
        recording_bridge=SimpleNamespace(service=service),
        analysis_eligibility_provider=lambda: (tuple(old_types),),
    )

    allowed = InputCalibration._can_start_recording_workflow(calibration)

    assert allowed is expected


def test_calibration_reports_capacity_backpressure_for_eligible_contexts():
    from ui.calibration_window import InputCalibration

    calibration = SimpleNamespace(
        recording_bridge=SimpleNamespace(
            service=SimpleNamespace(can_start_recording=False, busy=True)),
        analysis_eligibility_provider=lambda: (("SPL",),),
    )

    assert not InputCalibration._can_start_recording_workflow(calibration)


@pytest.mark.parametrize("analysis_type", [
    "ED", "PD", "PM", "FFT", "CQT", "AI", "future-analysis",
])
def test_request_scoped_background_analysis_rejects_out_of_scope_types(analysis_type):
    import numpy as np

    from ui.sequence.request_scoped_recording_analysis import analyze_recording_request

    request = SimpleNamespace(channels=(0,), calibration_metadata=None, device={})

    with pytest.raises(ValueError, match="SPL/FBA/SPEC"):
        analyze_recording_request(
            request=request,
            recorded_mono=np.ones(8, dtype=np.float32),
            recorded_multi=np.ones((8, 1), dtype=np.float32),
            sample_rate=48000,
            config_snapshot=_snapshot(analysis_type),
            recorded_signal_info={},
        )


def test_serialized_completion_uses_entry_frozen_config_after_host_mutation():
    import numpy as np

    from ui.sequence.sequence_widget_recording_process_ops import (
        SequenceWidgetRecordingProcessOpsMixin,
    )
    from ui.sequence.sequence_widget_streaming_ops import (
        SequenceWidgetStreamingOpsMixin,
    )

    frozen = _snapshot("ED")
    context = SimpleNamespace(
        publication_started=False,
        publication_delivered=False,
        publication_audio=SimpleNamespace(
            mono=np.ones(8, dtype=np.float32),
            multi=np.ones((8, 1), dtype=np.float32)),
        publication_sample_rate=48000,
        publication_final_windows=(),
        publication_attempts=0,
        recent_session_config_snapshot=frozen,
        cancelled=False,
        session=SimpleNamespace(),
        active_transition_applied=False,
    )
    observed = {}
    host = SimpleNamespace(
        analysis_config=_snapshot("PD")["analysis_config"],
        _notify_process_recording_finished=lambda *_args, **_kwargs: None,
        _finalize_recording_channel_selection=lambda: None,
        _drop_recording_context=lambda value: observed.setdefault("dropped", value),
    )

    def complete(**kwargs):
        observed["context"] = kwargs.get("recording_context")
        observed["serialized_legacy"] = kwargs.get("serialized_legacy")
        config = SequenceWidgetStreamingOpsMixin._recording_completion_analysis_config(
            host, kwargs.get("recording_context"))
        observed["type"] = config["item-0"]["type"]
        return True

    host._on_streaming_complete = complete

    assert SequenceWidgetRecordingProcessOpsMixin._publish_serialized_recording_context(
        host, context) is True
    assert observed == {
        "context": context,
        "serialized_legacy": True,
        "type": "ED",
        "dropped": context,
    }
    assert context.publication_audio is None
    assert context.publication_delivered is True
