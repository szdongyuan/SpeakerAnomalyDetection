import os

import numpy as np
import librosa

from consts import error_code, running_consts


def get_audio_files_and_labels(signal_path, sr=None, with_labels=-1, **kwargs):
    """
    Function to retrieve audio files and their corresponding labels from a directory.

    Args:
    - signal_dir (str): Directory containing audio files.
    - sr (int or None): Sampling rate for audio files. If None, uses default sampling rate.
    - with_labels (int): Label to assign to the audio files. Default is -1.

    Returns:
    - audio_signals (list): List containing audio signals loaded from files.
    - audio_file_names (list): List containing names of audio files.
    - fs (list): List containing sampling rates of audio files.
    - labels (list): List containing labels assigned to audio files.

    """
    audio_signals = []
    audio_file_names = []
    fs = []
    labels = []
    signal_path = signal_path.replace("\\", "/")
    if os.path.isfile(signal_path):
        signal_files = [os.path.basename(signal_path)]
        signal_path = os.path.dirname(signal_path)
    elif os.path.isdir(signal_path):
        signal_files = os.listdir(signal_path)
    else:
        return error_code.INVALID_PATH, "invalid path [%s]" % signal_path

    max_size = kwargs.get("max_size", len(signal_files))
    replace = True if max_size > len(signal_files) else False
    selected_files = np.random.choice(signal_files, size=max_size, replace=replace)
    for signal_file in selected_files:
        single_audio_path = os.path.join(signal_path, signal_file).replace("\\", "/")

        try:
            # AI training uses mono audio only
            y, sr = librosa.load(single_audio_path, sr=sr, mono=True)
            if fs and sr != fs[-1]:
                pass
            else:
                audio_signals.append(y)
                audio_file_names.append(signal_file)
                labels.append(with_labels)
                fs.append(sr)
        except Exception as e:
            print("something wrong")

    return error_code.OK, (audio_signals, audio_file_names, fs, labels)


def get_pre_labeled_audios_from_dict(pre_labeled_dict, **kwargs):
    audio_signals = []
    audio_file_names = []
    fs = []
    labels = []
    for key, value in pre_labeled_dict.items():
        path = running_consts.DEFAULT_DIR + key
        signal_file = os.path.basename(path)
        if "OK" == value[1]:
            label = 1
        else:
            label = 0
        try:
            # AI training uses mono audio only
            y, sr = librosa.load(path, sr=value[0], mono=True)
            if fs and sr != fs[-1]:
                pass
            else:
                audio_signals.append(y)
                audio_file_names.append(signal_file)
                labels.append(label)
                fs.append(sr)
        except Exception as e:
            print("something wrong")
    return error_code.OK, (audio_signals, np.array(audio_file_names), np.array(fs), np.array(labels))


def get_pre_labeled_audios(pre_labeled_dir, **kwargs):
    """
    Function to retrieve pre-labeled audio files from specified directories.

    Args:
    - pre_labeled_dir (str): Directory containing pre-labeled audio files.

    Returns:
    - tot_signals (ndarray): NumPy array containing concatenated audio signals.
    - tot_files (ndarray): NumPy array containing names of audio files.
    - tot_fs (ndarray): NumPy array containing sampling rates of audio files.
    - tot_labels (ndarray): NumPy array containing labels assigned to audio files.
    """
    if not os.path.isdir(pre_labeled_dir):
        return error_code.INVALID_PATH, "invalid directory [%s]" % pre_labeled_dir
    signal_dir = os.path.join(pre_labeled_dir, "%s").replace("\\", "/")

    load_kwargs = {}
    if kwargs.get("max_train_size"):
        load_kwargs["max_size"] = kwargs.get("max_train_size") // 2
    ret_code, ret = get_audio_files_and_labels(signal_dir % "OK", with_labels=1, **load_kwargs)
    if ret_code != error_code.OK:
        return ret_code, ret
    ok_signals, ok_files, ok_fs, ok_labels = ret
    ret_code, ret = get_audio_files_and_labels(signal_dir % "NG", with_labels=0, **load_kwargs)
    if ret_code != error_code.OK:
        return ret_code, ret
    ng_signals, ng_files, ng_fs, ng_labels = ret

    tot_signals = ok_signals + ng_signals
    tot_files = np.array(ok_files + ng_files)
    tot_fs = np.array(ok_fs + ng_fs)
    tot_labels = np.array(ok_labels + ng_labels)
    return error_code.OK, (tot_signals, tot_files, tot_fs, tot_labels)


def load_audio_simple(audio_path, sr=None):
    """
    Load audio file with multi-channel support.

    Args:
        audio_path (str): Path to audio file
        sr (int, optional): Target sample rate

    Returns:
        y (np.ndarray): Audio data
            - Shape (samples,) for mono
            - Shape (samples, channels) for multi-channel
        t (np.ndarray): Time array
    """
    if not audio_path:
        return None, None

    # Load audio with multi-channel support
    y, sr = librosa.load(audio_path, sr=sr, mono=False)

    # If y is 2D, transpose from (channels, samples) to (samples, channels)
    if y.ndim == 2:
        y = y.T

    t = np.linspace(0, len(y) - 1, len(y)) / sr if y.ndim == 1 else np.linspace(0, y.shape[0] - 1, y.shape[0]) / sr
    return y, t
