"""Main-thread-owned configuration state for one sequence window."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

import numpy as np

from base.data_struct.data_deal_struct import DataDealStruct
from ui.sequence.sequence_messages import ConfigurationSnapshot
from ui.sequence.sequence_recording_model import (
    RecordingModel,
    RecordingStreamingStimulusPort,
)


@dataclass(frozen=True, slots=True)
class ConfigurationStateCheckpoint:
    """Exact in-memory projection used to roll back a failed publication."""

    configuration_generation: int
    sequence_config: list[Any]
    analysis_config: dict[Any, Any]
    mic: Any
    speaker: Any
    mic_channels: list[int]
    using_config_path: Any
    streaming_stimulus_data: Any
    acquisition_mode: str | None
    workflow_configuration_generation: int | None
    workflow_configuration_snapshot: Any


def _mutable_payload(value: Any) -> Any:
    """Convert an admitted message snapshot back to legacy mutable containers."""
    if isinstance(value, Mapping):
        return {
            _mutable_payload(key): _mutable_payload(item)
            for key, item in value.items()
        }
    if type(value) is tuple:
        return [_mutable_payload(item) for item in value]
    if type(value) is frozenset:
        return {_mutable_payload(item) for item in value}
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, PurePath):
        return type(value)(value)
    return deepcopy(value)


class SequenceConfigurationModel:
    """Own configuration/device state while sharing generation with workflow state."""

    def __init__(
        self,
        *,
        data_struct: DataDealStruct | Any | None = None,
        workflow_model: Any | None = None,
        streaming_stimulus_port: RecordingStreamingStimulusPort | None = None,
    ) -> None:
        self._data_struct = data_struct if data_struct is not None else DataDealStruct()
        self._workflow_model = workflow_model
        self._streaming_stimulus_port = (
            streaming_stimulus_port
            if streaming_stimulus_port is not None
            else RecordingStreamingStimulusPort(RecordingModel())
        )
        self._configuration_generation = (
            int(workflow_model.configuration_generation)
            if workflow_model is not None
            else 0
        )
        self._mic: Any = None
        self._speaker: Any = None
        self._mic_channels: list[int] = []
        self._sequence_config: list[Any] = []
        self._analysis_config: dict[Any, Any] = {}
        self._using_config_path: Any = None
        self._registry: dict[Any, Any] = {}
        self._registry_entries: tuple[tuple[str, str], ...] = ()
        self._acquisition_mode: str | None = None
        self.audio_devices_available = True
        self.audio_devices_unavailable_message = ""
        self.stimulus_reference_ready = False

    @property
    def configuration_generation(self) -> int:
        if self._workflow_model is not None:
            return int(self._workflow_model.configuration_generation)
        return self._configuration_generation

    @property
    def data_struct(self) -> Any:
        return self._data_struct

    @property
    def mic(self) -> Any:
        return self._mic

    @mic.setter
    def mic(self, value: Any) -> None:
        self._mic = deepcopy(value)
        self._synchronize_workflow_snapshot()

    @property
    def speaker(self) -> Any:
        return self._speaker

    @speaker.setter
    def speaker(self, value: Any) -> None:
        self._speaker = deepcopy(value)
        self._synchronize_workflow_snapshot()

    @property
    def mic_channels(self) -> list[int]:
        return self._mic_channels

    @mic_channels.setter
    def mic_channels(self, value: Any) -> None:
        self._mic_channels = list(deepcopy(value) or [])
        self._synchronize_workflow_snapshot()

    @property
    def sequence_config(self) -> list[Any]:
        return self._sequence_config

    @sequence_config.setter
    def sequence_config(self, value: Any) -> None:
        normalized = _mutable_payload(value or [])
        self._sequence_config = normalized if isinstance(normalized, list) else []
        self._acquisition_mode = self._mode_from_sequence(self._sequence_config)
        self._synchronize_workflow_snapshot()

    @property
    def analysis_config(self) -> dict[Any, Any]:
        return self._analysis_config

    @analysis_config.setter
    def analysis_config(self, value: Any) -> None:
        normalized = _mutable_payload(value or {})
        self._analysis_config = normalized if isinstance(normalized, dict) else {}
        self._synchronize_workflow_snapshot()

    @property
    def using_config_path(self) -> Any:
        return self._using_config_path

    @using_config_path.setter
    def using_config_path(self, value: Any) -> None:
        self._using_config_path = deepcopy(value)
        self._synchronize_workflow_snapshot()

    @property
    def registry(self) -> dict[Any, Any]:
        return self._registry

    @property
    def registry_entries(self) -> tuple[tuple[str, str], ...]:
        return self._registry_entries

    @property
    def acquisition_mode(self) -> str | None:
        return self._acquisition_mode

    @acquisition_mode.setter
    def acquisition_mode(self, value: str | None) -> None:
        self._acquisition_mode = value

    @property
    def runtime_sample_rate(self) -> Any:
        return getattr(self._data_struct, "sample_rate", None)

    @runtime_sample_rate.setter
    def runtime_sample_rate(self, value: Any) -> None:
        self._data_struct.sample_rate = value

    @property
    def stimulus_reference_data(self) -> Any:
        return getattr(self._data_struct, "stimulus_data", None)

    @property
    def stimulus_reference_info(self) -> Any:
        return getattr(self._data_struct, "stimulus_info", None)

    @property
    def streaming_stimulus_data(self) -> Any:
        return self._streaming_stimulus_port.read()

    @streaming_stimulus_data.setter
    def streaming_stimulus_data(self, value: Any) -> None:
        self._streaming_stimulus_port.write(value)
        self._synchronize_workflow_snapshot()

    @staticmethod
    def _mode_from_sequence(sequence_config: list[Any]) -> str | None:
        try:
            return sequence_config[0]["seq1"]["acq"]["mode"]
        except (IndexError, KeyError, TypeError):
            return None

    def replace_registry(
        self,
        registry: Mapping[Any, Any] | None,
        *,
        using_config_path: Any,
        entries: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._registry = deepcopy(dict(registry or {}))
        self._using_config_path = deepcopy(using_config_path)
        self._registry_entries = tuple(
            (str(name), str(path)) for name, path in entries
        )

    def current_snapshot(self) -> ConfigurationSnapshot:
        return ConfigurationSnapshot(
            sequence_config=self._sequence_config,
            analysis_config=self._analysis_config,
            mic=self._mic,
            speaker=self._speaker,
            mic_channels=tuple(self._mic_channels),
            using_config_path=self._using_config_path,
            streaming_stimulus_data=self.streaming_stimulus_data,
        )

    def checkpoint_configuration_state(self) -> ConfigurationStateCheckpoint:
        workflow_generation = None
        workflow_snapshot = None
        if self._workflow_model is not None:
            workflow_generation = int(
                self._workflow_model.configuration_generation
            )
            workflow_snapshot = self._workflow_model.configuration_snapshot
        return ConfigurationStateCheckpoint(
            configuration_generation=self.configuration_generation,
            sequence_config=self._sequence_config,
            analysis_config=self._analysis_config,
            mic=self._mic,
            speaker=self._speaker,
            mic_channels=self._mic_channels,
            using_config_path=self._using_config_path,
            streaming_stimulus_data=self.streaming_stimulus_data,
            acquisition_mode=self._acquisition_mode,
            workflow_configuration_generation=workflow_generation,
            workflow_configuration_snapshot=workflow_snapshot,
        )

    def restore_configuration_state(
        self, checkpoint: ConfigurationStateCheckpoint
    ) -> None:
        if type(checkpoint) is not ConfigurationStateCheckpoint:
            raise TypeError(
                "checkpoint must be a ConfigurationStateCheckpoint"
            )
        self._configuration_generation = checkpoint.configuration_generation
        self._sequence_config = checkpoint.sequence_config
        self._analysis_config = checkpoint.analysis_config
        self._mic = checkpoint.mic
        self._speaker = checkpoint.speaker
        self._mic_channels = checkpoint.mic_channels
        self._using_config_path = checkpoint.using_config_path
        self._streaming_stimulus_port.write(checkpoint.streaming_stimulus_data)
        self._acquisition_mode = checkpoint.acquisition_mode
        if self._workflow_model is not None:
            self._workflow_model.configuration_generation = (
                checkpoint.workflow_configuration_generation
            )
            self._workflow_model.configuration_snapshot = (
                checkpoint.workflow_configuration_snapshot
            )

    def _synchronize_workflow_snapshot(self) -> None:
        if self._workflow_model is None:
            return
        self._workflow_model.apply_configuration(
            self.current_snapshot(),
            generation=self.configuration_generation,
        )

    def apply_configuration(
        self, snapshot: ConfigurationSnapshot, *, generation: int
    ) -> bool:
        if type(snapshot) is not ConfigurationSnapshot:
            raise TypeError("snapshot must be a ConfigurationSnapshot")
        if type(generation) is not int or generation < 0:
            raise ValueError("configuration generation must be a non-negative integer")
        if generation < self.configuration_generation:
            return False

        sequence_config = _mutable_payload(snapshot.sequence_config)
        analysis_config = _mutable_payload(snapshot.analysis_config)
        mic = _mutable_payload(snapshot.mic)
        speaker = _mutable_payload(snapshot.speaker)
        mic_channels = list(snapshot.mic_channels)
        using_config_path = _mutable_payload(snapshot.using_config_path)
        streaming_stimulus_data = _mutable_payload(snapshot.streaming_stimulus_data)
        if not isinstance(sequence_config, list):
            raise ValueError("sequence configuration snapshot must contain a list")
        if not isinstance(analysis_config, dict):
            raise ValueError("analysis configuration snapshot must contain a mapping")

        if self._workflow_model is not None and not self._workflow_model.apply_configuration(
            snapshot, generation=generation
        ):
            return False
        self._configuration_generation = generation
        self._sequence_config = sequence_config
        self._analysis_config = analysis_config
        self._mic = mic
        self._speaker = speaker
        self._mic_channels = mic_channels
        self._using_config_path = using_config_path
        self._streaming_stimulus_port.write(streaming_stimulus_data)
        self._acquisition_mode = self._mode_from_sequence(sequence_config)
        return True

    def set_audio_devices_available(self, available: bool, message: str = "") -> None:
        self.audio_devices_available = bool(available)
        self.audio_devices_unavailable_message = message or ""
