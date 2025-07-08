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
        t1 = np.linspace(0, t_time, num_samples)
        t2 = np.linspace(0, t_time, num_samples + 1)
        phase_position = 0
        start = 0
        end = 0
        for i in range(num_steps):
            if i == 0:
                time = t1
            else:
                time = t2
                start = end - 1
            end = start + len(time)
            fr = frequencies[i]
            y_t[start:end] = np.sin(2 * pi * fr * time + phase_position)
            phase_position = (phase_position + 2 * pi * fr * time[-1]) % (2 * pi)
        y_total = np.array(list(y_t) * repeat_times)
        return y_total, sample_rate

    @staticmethod
    def generate_chirps(start_freq=80, stop_freq=2000, total_time=4.0, repeat_times=1, sample_rate=44100,
                        stimulus_type="log", **kwargs):
        y_all = []
        t_single = total_time / repeat_times
        t_half = t_single / 2
        current_phase = 0.0
        for _ in range(repeat_times):
            if stimulus_type == "log" and start_freq != stop_freq:
                y_part, current_phase = StimulusSignal.make_sweep("log", start_freq, stop_freq, t_single,
                                                                  sample_rate, current_phase)
                y_all.append(y_part)
            elif stimulus_type == "linear" or start_freq == stop_freq:
                y_part, current_phase = StimulusSignal.make_sweep("linear", start_freq, stop_freq, t_single,
                                                                  sample_rate, current_phase)
                y_all.append(y_part)
            elif stimulus_type == "mirror_log":
                y_down, phase_end = StimulusSignal.make_sweep("log", stop_freq, start_freq, t_half,
                                                              sample_rate, current_phase)
                y_up, current_phase = StimulusSignal.make_sweep("log", start_freq, stop_freq, t_half,
                                                                sample_rate, phase_end)
                y_all.append(np.concatenate([y_down, y_up]))
            elif stimulus_type == "mirror_linear":
                y_down, phase_end = StimulusSignal.make_sweep("linear", stop_freq, start_freq, t_half,
                                                              sample_rate, current_phase)
                y_up, current_phase = StimulusSignal.make_sweep("linear", start_freq, stop_freq, t_half,
                                                                sample_rate, phase_end)
                y_all.append(np.concatenate([y_down, y_up]))
            else:
                raise Exception("Invalid chirp type")
        return np.concatenate(y_all), sample_rate

    @staticmethod
    def make_sweep(stimulus_type, freq1, freq2, duration, sr, phase_offset=0.0):
        t = np.arange(int(duration * sr)) / sr
        if stimulus_type == "linear":
            k = (freq2 - freq1) / duration
            phase = 2 * np.pi * (0.5 * k * t ** 2 + freq1 * t)
        elif stimulus_type == "log":
            ln_ratio = np.log(freq2 / freq1)
            phase = 2 * np.pi * freq1 * duration / ln_ratio * (np.exp(ln_ratio * t / duration) - 1)
        else:
            raise Exception("Invalid chirp type")
        y = phase + phase_offset
        return np.sin(y), y[-1]

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
