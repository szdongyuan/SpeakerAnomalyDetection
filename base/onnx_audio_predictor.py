import math
import os

import librosa
import numpy as np
import onnxruntime as ort

from base.load_config import load_config


class OnnxAudioPredictor:
    class_names = ["OK", "NG"]

    def __init__(self, model_path: str, config: dict):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = model_path
        self.config = config or {}

        data_config = self.config.get("data", {})
        self.cqt_config = self.config.get("cqt", {})
        if not self.cqt_config:
            raise ValueError("Missing 'cqt' configuration for ONNX inference.")

        self.sr = int(data_config.get("sr", 44100))
        self.duration = float(data_config.get("duration", 4.0))
        self.expected_num_samples = int(round(self.sr * self.duration))
        self.acc_req = self._get_acc_req()

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        self.input_name = self.input_info.name
        self.output_name = self.output_info.name
        self.input_shape = list(self.input_info.shape)

        self.target_n_bins = self._resolve_shape_axis(
            self.input_shape[-2] if len(self.input_shape) >= 2 else None,
            fallback=int(self.cqt_config.get("n_bins", 640)),
        )
        self.target_time_frames = self._resolve_shape_axis(
            self.input_shape[-1] if len(self.input_shape) >= 1 else None,
            fallback=self._default_time_frames(),
        )

    def _get_acc_req(self) -> float:
        inference_config = self.config.get("inference", {})
        predict_config = self.config.get("model_predict_config", {})
        return float(inference_config.get("acc_req", predict_config.get("acc_req", 0.5)))

    def _default_time_frames(self) -> int:
        hop_length = max(int(self.cqt_config.get("hop_length", 128)), 1)
        return int(math.ceil(self.expected_num_samples / hop_length))

    @staticmethod
    def _resolve_shape_axis(value, fallback: int) -> int:
        if isinstance(value, int) and value > 0:
            return value
        return fallback

    def _prepare_audio(self, audio: np.ndarray, src_sr: int) -> np.ndarray:
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("Audio data is empty.")

        if src_sr and int(src_sr) != self.sr:
            waveform = librosa.resample(waveform, orig_sr=int(src_sr), target_sr=self.sr)

        if self.expected_num_samples > 0:
            if waveform.size < self.expected_num_samples:
                pad_width = self.expected_num_samples - waveform.size
                waveform = np.pad(waveform, (0, pad_width), mode="constant")
            elif waveform.size > self.expected_num_samples:
                waveform = waveform[:self.expected_num_samples]

        return waveform.astype(np.float32, copy=False)

    def compute_cqt(self, audio: np.ndarray) -> np.ndarray:
        cqt = librosa.cqt(
            audio,
            sr=self.sr,
            n_bins=int(self.cqt_config.get("n_bins", self.target_n_bins)),
            bins_per_octave=int(self.cqt_config.get("bins_per_octave", 64)),
            fmin=float(self.cqt_config.get("fmin", 20.5)),
            hop_length=int(self.cqt_config.get("hop_length", 128)),
            pad_mode="constant",
        )

        cqt_mag = np.abs(cqt)
        ref_value = float(np.max(cqt_mag)) if cqt_mag.size else 1.0
        if (not np.isfinite(ref_value)) or ref_value <= 0:
            ref_value = 1.0
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=ref_value)

        cqt_norm = (cqt_db + 80.0) / 80.0
        cqt_norm = np.clip(cqt_norm, -1.0, 1.0)

        if cqt_norm.shape[0] != self.target_n_bins:
            if cqt_norm.shape[0] < self.target_n_bins:
                bin_pad = self.target_n_bins - cqt_norm.shape[0]
                cqt_norm = np.pad(cqt_norm, ((0, bin_pad), (0, 0)), mode="constant")
            else:
                cqt_norm = cqt_norm[: self.target_n_bins, :]

        if cqt_norm.shape[1] < self.target_time_frames:
            frame_pad = self.target_time_frames - cqt_norm.shape[1]
            cqt_norm = np.pad(cqt_norm, ((0, 0), (0, frame_pad)), mode="constant")
        elif cqt_norm.shape[1] > self.target_time_frames:
            cqt_norm = cqt_norm[:, : self.target_time_frames]

        return cqt_norm[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _to_probabilities(self, outputs: np.ndarray) -> np.ndarray:
        scores = np.asarray(outputs, dtype=np.float32)
        if scores.ndim == 1:
            scores = scores[np.newaxis, :]

        row_sums = np.sum(scores, axis=1)
        if np.all(scores >= 0.0) and np.all(scores <= 1.0) and np.allclose(row_sums, 1.0, atol=1e-3):
            return scores
        return self._softmax(scores)

    def predict_array(self, audio: np.ndarray, sr: int, file_name: str = "") -> dict:
        prepared_audio = self._prepare_audio(audio, sr)
        cqt_features = self.compute_cqt(prepared_audio)
        outputs = self.session.run([self.output_name], {self.input_name: cqt_features})[0]
        probabilities = self._to_probabilities(outputs)

        predicted_label = int(np.argmax(probabilities, axis=1)[0])
        return {
            "file_name": file_name,
            "predicted_class": self.class_names[predicted_label],
            "predicted_label": predicted_label,
            "confidence": float(probabilities[0][predicted_label]),
            "probabilities": {
                "OK": float(probabilities[0][0]),
                "NG": float(probabilities[0][1]),
            },
        }

    def predict_arrays(self, signals, file_names, fs):
        results = []
        for idx, signal in enumerate(signals):
            result = self.predict_array(signal, fs[idx], file_name=file_names[idx])
            results.append(result)
        return results


def build_onnx_model_summary(model_path: str, config_path: str) -> str:
    config = load_config(config_path=config_path)
    data_config = config.get("data", {}) if isinstance(config, dict) else {}
    cqt_config = config.get("cqt", {}) if isinstance(config, dict) else {}
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    inference_config = config.get("inference", {}) if isinstance(config, dict) else {}

    sr = int(data_config.get("sr", 44100))
    duration = float(data_config.get("duration", 4.0))
    hop_length = max(int(cqt_config.get("hop_length", 128)), 1)
    time_frames = int(math.ceil((sr * duration) / hop_length))
    n_bins = int(cqt_config.get("n_bins", 640))
    acc_req = float(inference_config.get("acc_req", 0.5))

    summary_lines = [
        "Model: ONNX Runtime (CPU)",
        f"Model Path: {model_path}",
        f"Config Path: {config_path}",
        f"Original Training Model: {model_config.get('model_name', 'Unknown')}",
        f"Sample Rate: {sr}",
        f"Duration: {duration:.2f}s",
        f"Expected Raw Input: {int(round(sr * duration))} x 1",
        f"CQT Input Tensor: 1 x 1 x {n_bins} x {time_frames}",
        f"CQT bins_per_octave: {cqt_config.get('bins_per_octave', 64)}",
        f"CQT fmin: {cqt_config.get('fmin', 20.5)}",
        f"CQT hop_length: {hop_length}",
        f"Classes: {OnnxAudioPredictor.class_names}",
        f"OK Threshold: {acc_req:.2f}",
    ]
    return "\n".join(summary_lines)
