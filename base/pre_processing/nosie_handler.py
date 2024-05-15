from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class NoiseHandler:
    def __init__(self, directory_path):
        self.directory_path = Path(directory_path)

    def process_selected_audios(self, n=3):
        """加载所有 WAV files"""
        audio_data = []
        audio_files = list(self.directory_path.glob('*.wav'))
        for i in range(min(n, len(audio_files))):
            y, sr = librosa.load(audio_files[i], sr=None)
            audio_data.append((audio_files[i].name, sr, y))
        return audio_data

    @staticmethod
    def plot_waveform_and_info(file_path):
        """可视化"""
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
        """保存 modified audio"""
        sf.write(file_path, audio, sr)

    def sample_random_noise(self, wave_data_list, num_samples=10, sample_length=64340):
        """随机截取噪声."""
        random_noise_samples = []
        if len(wave_data_list) < num_samples:
            print("Not enough audio files to sample from.")
            return []

        for i in range(num_samples):
            file_name, sr, y = wave_data_list[i]
            start = np.random.randint(0, len(y) - sample_length)
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
        """添加 random factory noise 并保存"""
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

