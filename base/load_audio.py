import os

import numpy as np
import librosa
import librosa.feature.spectral as spectral

from consts import error_code


def get_audio_files_and_labels(signal_path,
                               sr=None,
                               with_labels=-1):
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

    for signal_file in signal_files:
        single_audio_path = os.path.join(signal_path, signal_file).replace("\\", "/")

        try:
            y, sr = librosa.load(single_audio_path, sr=sr)
            if fs and sr != fs[-1]:
                pass
            else:
                audio_signals.append(y)
                audio_file_names.append(signal_file)
                labels.append(with_labels)
                fs.append(sr)
        except Exception as e:
            print("something wrong")

    return error_code.OK, (np.array(audio_signals), audio_file_names, fs, labels)


def get_pre_labeled_audios(pre_labeled_dir):
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

    ret_code, ret = get_audio_files_and_labels(signal_dir % "OK", with_labels=1)
    if ret_code != error_code.OK:
        return ret_code, ret
    ok_signals, ok_files, ok_fs, ok_labels = ret
    ret_code, ret = get_audio_files_and_labels(signal_dir % "NG", with_labels=0)
    if ret_code != error_code.OK:
        return ret_code, ret
    ng_signals, ng_files, ng_fs, ng_labels = ret

    tot_signals = np.vstack((ok_signals, ng_signals))
    tot_files = np.array(ok_files + ng_files)
    tot_fs = np.array(ok_fs + ng_fs)
    tot_labels = np.array(ok_labels + ng_labels)
    return error_code.OK, (tot_signals, tot_files, tot_fs, tot_labels)


def pre_process_data(signals, preprocess_config, **kwargs):
    """
        Function to pre-process audio signals using the specified method.

        Args:
        - signals (list): List containing audio signals.
        - method (str): Method for pre-processing. Default is "mfcc".
        - **kwargs: Additional keyword arguments to be passed to the pre-processing function.

        Returns:
        - processed_data (ndarray): NumPy array containing pre-processed data.
    """
    method = preprocess_config.get("preprocess_method", "mfcc")
    preprocess_kwargs = preprocess_config.get("preprocess_param", {})
    if method == "mfcc":
        processed_data = []
        fs = kwargs.get("fs")
        for i in range(len(signals)):
            mfcc = spectral.mfcc(y=signals[i], sr=fs[i], **preprocess_kwargs).T
            if preprocess_config.get("data_reshape", True):
                mfcc = mfcc.reshape((1, mfcc.shape[0] * mfcc.shape[1]))[0]
            processed_data.append(mfcc)
        return np.array(processed_data)
    elif method == "mel_spec":
        processed_data = []
        fs = kwargs.get("fs")
        for i in range(len(signals)):
            mel_spec = spectral.melspectrogram(y=signals[i], sr=fs[i], **preprocess_kwargs).T
            if preprocess_config.get("data_reshape", True):
                mel_spec = mel_spec.reshape((1, mel_spec.shape[0] * mel_spec.shape[1]))[0]
            processed_data.append(mel_spec)
        return np.array(processed_data)
    else:
        print("method [%s] not support yet" % method)
        return signals
