import numpy as np
import matplotlib.pyplot as plt


class stimulus(object):
    @staticmethod
    def stepped_sine_sweep(f_start, f_end, steps, duration_per_step, sampling_rate):
        """
        Stepped Sine Sweep

        Parameters:
            f_start (float): Starting frequency (Hz).
            f_end (float): Ending frequency (Hz).
            steps (int): Number of steps in the frequency sweep.
            duration_per_step (float): Duration of each frequency step (seconds).
            sampling_rate (int): Sampling rate (Hz).

        Returns:
            t (ndarray): Time array.
            sweep_signal (ndarray): Stepped sine sweep signal.
        """

        # Time step and total duration
        t_step = 1 / sampling_rate
        t_total = steps * duration_per_step
        t = np.arange(0, t_total, t_step)

        frequencies = np.linspace(f_start, f_end, steps)

        sweep_signal = np.zeros_like(t)

        current_idx = 0
        for f in frequencies:
            cycles = int(f * duration_per_step)

            t_step_signal = np.arange(0, duration_per_step, t_step)
            sine_wave = np.sin(2 * np.pi * f * t_step_signal)

            step_size = len(t_step_signal)
            sweep_signal[current_idx:current_idx + step_size] = sine_wave

            current_idx += step_size

        return t, sweep_signal

    @staticmethod
    def log_sweep(f_start, f_end, duration, sampling_rate):
        """
        Frequency Log Sweep

        Parameters:
            f_start (float): Starting frequency (Hz).
            f_end (float): Ending frequency (Hz).
            duration (float): Total duration of the sweep (seconds).
            sampling_rate (int): Sampling rate (Hz).

        Returns:
            t (ndarray): Time array.
            sweep_signal (ndarray): Logarithmic sine sweep signal.
        """

        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

        K = np.log(f_end / f_start) / duration
        instantaneous_frequency = f_start * np.exp(K * t)

        sweep_signal = np.sin(2 * np.pi * instantaneous_frequency * t)

        return t, sweep_signal

    @staticmethod
    def log_amplitude_sweep(frequency, duration, sweep_rate, sampling_rate):
        """
        Log Amplitude Sweep

        Parameters:
            frequency (float): Constant frequency (Hz).
            duration (float): Total duration of the sweep (seconds).
            sweep_rate (float): Rate of amplitude change (time per decade, seconds).
            sampling_rate (int): Sampling rate (Hz).

        Returns:
            t (ndarray): Time array.
            sweep_signal (ndarray): Logarithmic amplitude sweep signal.
        """

        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

        amp = np.exp((t / sweep_rate) * np.log(10))

        sine_wave = np.sin(2 * np.pi * frequency * t)

        sweep_signal = amp * sine_wave

        return t, sweep_signal

    @staticmethod
    def two_tone_signal(f1, f2, duration, sampling_rate, amp1=1.0, amp2=1.0):
        """
        Two-Tone Signal

        Parameters:
            f1 (float): First frequency (Hz).
            f2 (float): Second frequency (Hz).
            duration (float): Signal duration (seconds).
            sampling_rate (int): Sampling rate (Hz).
            amp1 (float): Amplitude of the first tone.
            amp2 (float): Amplitude of the second tone.

        Returns:
            t (ndarray): Time array.
            signal (ndarray): Generated two-tone signal.
        """

        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

        tone1 = amp1 * np.sin(2 * np.pi * f1 * t)
        tone2 = amp2 * np.sin(2 * np.pi * f2 * t)

        signal = tone1 + tone2

        return t, signal

    @staticmethod
    def schroeder_multitone(f_start, f_end, n_tones, duration, sampling_rate):
        """
        Multitone Signal with Schroeder Phase Optimization

        Parameters:
            f_start (float): Starting frequency (Hz).
            f_end (float): Ending frequency (Hz).
            n_tones (int): Number of tones in the multitone signal.
            duration (float): Signal duration (seconds).
            sampling_rate (int): Sampling rate (Hz).

        Returns:
            t (ndarray): Time array.
            signal (ndarray): Generated multitone signal.
        """

        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

        frequencies = np.linspace(f_start, f_end, n_tones)

        phases = np.array([(np.pi * k * (k - 1)) / n_tones for k in range(n_tones)])

        signal = np.zeros_like(t)
        for i, f in enumerate(frequencies):
            signal += np.sin(2 * np.pi * f * t + phases[i])

        signal /= n_tones

        return t, signal


f_start = 20
f_end = 2000
steps = 50
duration = 2
duration_per_step = 0.1
sampling_rate = 44100
frequency = 1000
sweep_rate = 0.1
amp1 = 1.0
amp2 = 0.8
n_tones = 10

t, sweep_signal = stimulus.stepped_sine_sweep(f_start, f_end, steps, duration_per_step, sampling_rate)

t1, sweep_signal1 = stimulus.log_sweep(f_start, f_end, duration, sampling_rate)

t2, sweep_signal2 = stimulus.log_amplitude_sweep(frequency, duration, sweep_rate, sampling_rate)

t3, sweep_signal3 = stimulus.two_tone_signal(f_start, f_end, duration, sampling_rate, amp1, amp2)

t4, sweep_signal4 = stimulus.schroeder_multitone(f_start, f_end, n_tones, duration, sampling_rate)

plt.figure(figsize=(10, 6))
plt.plot(t4, sweep_signal4)
plt.title('Stepped Sine Sweep Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
