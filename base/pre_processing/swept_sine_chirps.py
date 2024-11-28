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
    def generate_chirps(start_freq=80, stop_freq=2000,
                        total_time=2.0, sample_rate=48000, repeat_times=1,
                        stimulus_type="log",
                        **kwargs):
        pi = np.pi
        t_single = total_time / repeat_times
        x_t = np.array(list(range(int(sample_rate * t_single)))) / sample_rate
        if stimulus_type == "log":
            ln = np.log(stop_freq / start_freq)
            y_t = np.sin(2 * pi * start_freq * t_single / ln * (np.exp(ln * x_t / t_single) - 1))
        elif stimulus_type == "linear":
            delta_f = stop_freq - start_freq
            y_t = np.sin(2 * pi * (0.5 * delta_f / t_single * x_t ** 2 + start_freq * x_t))
        elif stimulus_type == "mirror_log":
            y_part, _ = StimulusSignal().generate_chirps(start_freq=start_freq, stop_freq=stop_freq,
                                                         total_time=t_single / 2, sample_rate=sample_rate,
                                                         stimulus_type="log")
            y_t = np.array(list(y_part)[::-1] + list(-1 * y_part))
        elif stimulus_type == "mirror_linear":
            y_part, _ = StimulusSignal().generate_chirps(start_freq=start_freq, stop_freq=stop_freq,
                                                         total_time=t_single / 2, sample_rate=sample_rate,
                                                         stimulus_type="linear")
            y_t = np.array(list(y_part)[::-1] + list(-1 * y_part))
        else:
            raise Exception("Invalid chirp type")
        y_total = np.array(list(y_t) * repeat_times)
        return y_total, sample_rate

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
            freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)
            freqs[0] = 1
            fft /= np.sqrt(freqs)
            y_t = np.fft.irfft(fft, n=num_samples)
        else:
            raise Exception("Invalid noise type.")
        y_total = np.array(list(y_t) * repeat_times)
        return y_total, sample_rate
