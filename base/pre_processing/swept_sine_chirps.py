import numpy as np


class StimulusSignal(object):
    @staticmethod
    def generate_steps(start_freq=500, stop_freq=1000,
                       total_time=4.0, sample_rate=44100, num_steps=10, repeat_times=1,
                       stimulus_type="linear",
                       **kwargs):
        if total_time == 0:
            return [], sample_rate
        t_single = total_time / repeat_times
        t_time = t_single / num_steps
        num_samples = int(t_time * sample_rate)
        pi = np.pi
        if stimulus_type == 'linear':
            frequencies = np.linspace(start_freq, stop_freq, num_steps)
        elif stimulus_type == 'log':
            frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), num_steps)
        else:
            raise Exception("Invalid step type.")
        y_t = np.zeros(num_samples * num_steps)
        sample_offsets = np.arange(num_samples, dtype=float)
        phase_position = 0.0
        for i, fr in enumerate(frequencies):
            start = i * num_samples
            end = start + num_samples
            y_t[start:end] = np.sin(phase_position + 2 * pi * float(fr) * sample_offsets / sample_rate)
            phase_position = (phase_position + 2 * pi * float(fr) * num_samples / sample_rate) % (2 * pi)
        y_total = np.array(list(y_t) * repeat_times)
        return y_total, sample_rate

    @staticmethod
    def generate_chirps(start_freq=80, stop_freq=2000, total_time=4.0, repeat_times=1, sample_rate=44100,
                        stimulus_type="log", **kwargs):
        y_all = []
        t_single = total_time / repeat_times
        t_half = t_single / 2
        current_phase = 0.0
        for i in range(repeat_times):
            add_point = True if i > 0 else False
            if stimulus_type == "log" and start_freq != stop_freq:
                y_part, current_phase = StimulusSignal.make_sweep("log", start_freq, stop_freq, t_single,
                                                                  sample_rate, current_phase, add_point=add_point)
                y_all.append(y_part)
            elif stimulus_type == "linear" or stimulus_type == "log":
                y_part, current_phase = StimulusSignal.make_sweep("linear", start_freq, stop_freq, t_single,
                                                                  sample_rate, current_phase, add_point=add_point)
                y_all.append(y_part)
            elif stimulus_type == "mirror_log":
                y_down, phase_end = StimulusSignal.make_sweep("log", stop_freq, start_freq, t_half,
                                                              sample_rate, current_phase, add_point=add_point)
                y_up, current_phase = StimulusSignal.make_sweep("log", start_freq, stop_freq, t_half,
                                                                sample_rate, phase_end, add_point=True)
                y_all.append(np.concatenate([y_down, y_up]))
            elif stimulus_type == "mirror_linear":
                y_down, phase_end = StimulusSignal.make_sweep("linear", stop_freq, start_freq, t_half,
                                                              sample_rate, current_phase, add_point=add_point)
                y_up, current_phase = StimulusSignal.make_sweep("linear", start_freq, stop_freq, t_half,
                                                                sample_rate, phase_end, add_point=True)
                y_all.append(np.concatenate([y_down, y_up]))
            else:
                raise Exception("Invalid chirp type")
        return np.concatenate(y_all), sample_rate

    @staticmethod
    def make_sweep(stimulus_type, freq1, freq2, duration, sr, phase_offset=0.0, add_point=False):
        num_samples = int(duration * sr)
        if add_point:
            num_samples += 1
        t = np.linspace(0, duration, num_samples, endpoint=True)
        if stimulus_type == "linear":
            k = (freq2 - freq1) / duration
            phase = 2 * np.pi * (0.5 * k * t ** 2 + freq1 * t)
        elif stimulus_type == "log":
            ln_ratio = np.log(freq2 / freq1)
            phase = 2 * np.pi * freq1 * duration / ln_ratio * (np.exp(ln_ratio * t / duration) - 1)
        else:
            raise Exception("Invalid chirp type")
        y = phase + phase_offset
        signal = np.sin(y)
        if add_point:
            signal = signal[1:]
        return signal, y[-1]

    @staticmethod
    def generate_noise(total_time=2.0, sample_rate=44100, repeat_times=1,
                       stimulus_type='white_noise',
                       **kwargs):
        t_single = total_time / repeat_times
        x_t = np.array(list(range(int(sample_rate * t_single)))) / sample_rate
        num_samples = len(x_t)
        if stimulus_type == 'white_noise':
            y_t = np.random.normal(0, 1, num_samples)
        elif stimulus_type == 'pink_noise':
            white_noise = np.random.normal(0, 1, num_samples)
            fft = np.fft.rfft(white_noise)
            freqs = np.fft.rfftfreq(num_samples, d=1 / sample_rate)
            freqs[0] = 1
            fft /= np.sqrt(freqs)
            y_t = np.fft.irfft(fft, n=num_samples)
        else:
            raise Exception("Invalid noise type.")
        y_total = np.array(list(y_t) * repeat_times)
        return y_total, sample_rate
