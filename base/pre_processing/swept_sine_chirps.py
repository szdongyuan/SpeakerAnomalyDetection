import numpy as np


class StimulusSignal(object):
    @staticmethod
    def generate_steps(f_begin=500, f_end=1000, t_total=4.0, sr=44100, num_steps=10, step_type="linear", repeat=1):
        t_single = t_total / repeat
        t_time = t_single / num_steps
        num_samples = int(t_time * sr)
        pi = np.pi
        if step_type == 'linear':
            frequencies = np.linspace(f_begin, f_end, num_steps)
        elif step_type == 'log':
            frequencies = np.logspace(np.log10(f_begin), np.log10(f_end), num_steps)
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
        y_total = np.array(list(y_t) * repeat)
        return y_total, sr

    @staticmethod
    def generate_chirps(f_begin=80, f_end=2000, t_total=2.0, sr=48000, chirp_type="log", repeat=1):
        pi = np.pi
        t_single = t_total / repeat
        x_t = np.array(list(range(int(sr * t_single)))) / sr
        if chirp_type == "log":
            ln = np.log(f_end / f_begin)
            y_t = np.sin(2 * pi * f_begin * t_single / ln * (np.exp(ln * x_t / t_single) - 1))
        elif chirp_type == "linear":
            delta_f = f_end - f_begin
            y_t = np.sin(2 * pi * (0.5 * delta_f / t_single * x_t ** 2 + f_begin * x_t))
        elif chirp_type == "mirror_log":
            y_part, _ = StimulusSignal().generate_chirps(f_begin, f_end, t_single / 2, sr, "log")
            y_t = np.array(list(y_part)[::-1] + list(-1 * y_part))
        elif chirp_type == "mirror_linear":
            y_part, _ = StimulusSignal().generate_chirps(f_begin, f_end, t_single / 2, sr, "linear")
            y_t = np.array(list(y_part)[::-1] + list(-1 * y_part))
        else:
            raise Exception("Invalid chirp type")
        y_total = np.array(list(y_t) * repeat)
        return y_total, sr

    @staticmethod
    def generate_noise(t_total=2.0, sr=44100, noise_type='white', repeat=1):
        t_single = t_total / repeat
        x_t = np.array(list(range(int(sr * t_single)))) / sr
        num_samples = len(x_t)
        if noise_type == 'white':
            y_t = np.random.normal(0, 1, num_samples)
        elif noise_type == 'pink':
            white_noise = np.random.normal(0, 1, num_samples)
            fft = np.fft.rfft(white_noise)
            freqs = np.fft.rfftfreq(num_samples, d=1/sr)
            freqs[0] = 1
            fft /= np.sqrt(freqs)
            y_t = np.fft.irfft(fft, n=num_samples)
        else:
            raise Exception("Invalid noise type.")
        y_total = np.array(list(y_t) * repeat)
        return y_total, sr
