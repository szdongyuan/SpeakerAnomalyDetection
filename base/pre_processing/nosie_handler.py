from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class NoiseHandler:
    def __init__(self, directory_path):
        """Initializes the NoiseHandler class with the directory path that contains the audio files."""
        self.directory_path = Path(directory_path)

    def process_selected_audios(self, n=3):
        """
            Load the specified number of WAV files from the directory.
            Args:
            - n: int
                Number of audio files to be processed.

            Returns:
            - audio_data: list
                The audio data list with file name, original sampling rate, audio data.
        """
        audio_data = []
        audio_files = list(self.directory_path.glob('*.wav')) # Get all WAV files in the directory
        for i in range(min(n, len(audio_files))):
            y, sr = librosa.load(audio_files[i], sr=None) # Load the audio file with original sample rate
            audio_data.append((audio_files[i].name, sr, y))
        return audio_data

    @staticmethod
    def plot_waveform_and_info(file_path):
        """
            Plots the waveform of the audio file and prints its sample rate and duration.

            Args:
            - file_path: string, int, pathlib.Path, soundfile.SoundFile, audioread object, or file-like object
                The path to the audio file.
        """
        y, sr = librosa.load(file_path, sr=None)
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()
        plt.close()
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"File: {file_path}")
        print(f"Sample Rate: {sr} Hz")
        print(f"Duration: {duration:.2f} seconds")

    @staticmethod
    def save_audio(file_path, audio, sr):
        """save modified audio"""
        sf.write(file_path, audio, sr)

    def sample_random_noise(self, wave_data_list, num_samples=10, sample_length=64340):
        """
            Sample random noise sample data from provided audio data.

            Args:
            - wave_data_list: list
                The audio data list with file name, original sampling rate, audio data.
            - num_samples: int
                The number of random samples to be extracted.
            - sample_length: int
                The length of random samples extracted.

            Returns:
            - random_noise_samples: list
                A tuple list of audio data file names, sample rates, and random noise sample data.
        """

        random_noise_samples = []
        if len(wave_data_list) < num_samples:
            print("Not enough audio files to sample from.")
            return []

        for i in range(num_samples):
            file_name, sr, y = wave_data_list[i]
            start = np.random.randint(0, len(y) - sample_length) # Sets the random starting index of the sample
            y_random_sample = y[start:start + sample_length]
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y_random_sample, sr=sr)
            plt.title(f"Random {sample_length} points from {i}: {file_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.show()
            plt.close()
            random_noise_samples.append((file_name, sr, y_random_sample))
            print(f"File {i}: {file_name}")
            print(f"Sample Rate: {sr} Hz")
            print(f"Total Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
            print(f"Random Sample Start Index: {start}")
        return random_noise_samples

    def add_factory_noise(self, target_audio_data, random_noise_samples, output_folder):
        """
            Adds random noise sample data to the target audio data and saves the result.

            Args:
            - target_audio_data: list
                A tuple list of audio data file names, sample rate, and audio data of the target audio.
            - random_noise_samples: list
                A tuple list of audio data file names, sample rates, and random noise sample data.
            - output_folder: string, int, pathlib.Path, soundfile.SoundFile, audioread object, or file-like object
                The path to the folder where the combined audio files are saved.
        """

        output_folder = Path(output_folder)
        for file_name, target_sr, target_audio in target_audio_data:
            base_name = file_name.split('.')[0]
            for i, (noise_file, noise_sr, noise_sample) in enumerate(random_noise_samples):
                if target_sr != noise_sr:
                    noise_sample = librosa.resample(noise_sample, orig_sr=noise_sr, target_sr=target_sr)

                min_length = min(len(target_audio), len(noise_sample))
                combined_audio = target_audio[:min_length] + noise_sample[:min_length]
                output_path = output_folder / f"{base_name}_combined_{i}_{noise_file}"
                self.save_audio(output_path, combined_audio, target_sr)

