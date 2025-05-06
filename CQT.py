import librosa
import numpy as np
import matplotlib.pyplot as plt



y, sr = librosa.load('audio_data/test/OK/S004-1_2024-12-25_107c610bb999_030.wav', sr=None)

def compute_cqt(y, sr=44100, hop_length=128, n_fft=1024, fmin=None, fmax=None, bins_per_octave=None, n_bins=None):
    """
    Compute the Constant-Q Transform (CQT) of an audio signal.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : number > 0
        Sampling rate of y
    hop_length : int > 0
        Number of samples between frames
    n_fft : int > 0
        FFT window size, used to determine frequency resolution
    fmin : float > 0
        Minimum frequency
    fmax : float > 0
        Maximum frequency. If None, defaults to sr/3
    bins_per_octave : int > 0 or None
        Number of bins per octave. If None, calculated based on n_fft.
    n_bins : int > 0 or None
        Total number of CQT bins. If None, calculated based on fmin, fmax and bins_per_octave.
        
    Returns
    -------
    CQT : np.ndarray
        Constant-Q transform of y
    C_mag : np.ndarray
        Magnitude of Constant-Q transform
    freqs : np.ndarray
        Frequencies corresponding to each bin of CQT
    times : np.ndarray
        Time points corresponding to each frame of CQT
    """
    
    if fmin is None:
        fmin = librosa.note_to_hz('C1')  # 32.7 Hz
    
    if fmax is None:
        fmax = librosa.note_to_hz('C9')
    
    if bins_per_octave is None:
        ## 表示每个八度内有多少频率点，对应的频率对数增加, 增加n_fft会增加频率分辨率，和预期相符
        bins_per_octave = int(12 * np.log2(n_fft/1024) + 24)  
        bins_per_octave = max(12, bins_per_octave)  
    
    
    if n_bins is None:
        n_octaves =  np.log2(fmax / fmin)    ## 八度，表示频率区间跨越了多少频率翻倍的区间
        n_bins = int(np.ceil(n_octaves * bins_per_octave))   ## n_bins ≈ bins_per_octave * log2(fmax/fmin)，最终的采样点数
    
    # fmax parameter is not supported in librosa.cqt
    C = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    )
    
    # Convert to magnitude
    # C_mag = np.abs(C)
    
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    times = librosa.times_like(C, sr=sr, hop_length=hop_length)
    # 注意 C 是复数
    print(freqs)
    return C, freqs, times

# Example usage
if __name__ == "__main__":
    # Load audio file if it hasn't been loaded already
    try:
        # Check if y and sr are already defined
        len(y)
    except NameError:
        # If not, load the audio file
        y, sr = librosa.load('audio_data/test/OK/S004-1_2024-12-25_107c610bb999_030.wav', sr=None)
    
    print(f"Audio loaded: {len(y)/sr:.2f} seconds, {sr} Hz sample rate")
    
    # Compute CQT
    C, freqs, times = compute_cqt(y, sr=sr)
    C_mag = np.abs(C)
    # Display frequency and time information
    print(f"Frequency range: {freqs.min():.1f} Hz to {freqs.max():.1f} Hz")
    print(f"Number of frequency bins: {len(freqs)}")
    print(f"Time range: {times.min():.2f}s to {times.max():.2f}s")
    print(f"Number of time frames: {len(times)}")
    
    # Print first few frequencies to check their distribution
    print("\nFirst 10 frequency bins (Hz):")
    for i, f in enumerate(freqs[:10]):
        print(f"Bin {i}: {f:.2f} Hz")
    
    print("\nSample of bins across spectrum:")
    for i in range(0, len(freqs), len(freqs)//10):
        if i < len(freqs):
            print(f"Bin {i}: {freqs[i]:.2f} Hz")
    
    # Plot with manually controlled frequency axis
    plt.figure(figsize=(12, 6))
    
    # Using pcolormesh to have more control over axis
    plt.pcolormesh(times, freqs, librosa.amplitude_to_db(C_mag, ref=np.max), 
                   shading='auto', cmap='viridis')
    
    # Set logarithmic scale for frequency axis to match CQT's logarithmic nature
    plt.yscale('log')
    
    # Let's use the actual frequency values to set the ticks
    # Find reasonable tick values
    tick_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    tick_values = [v for v in tick_values if freqs.min() <= v <= freqs.max()]
    
    if len(tick_values) == 0:
        # If no standard values in our range, use min, max and some points in between
        tick_values = np.linspace(freqs.min(), freqs.max(), 5)
    
    tick_labels = []
    for v in tick_values:
        if v >= 1000:
            tick_labels.append(f"{v/1000:.1f}k")
        else:
            tick_labels.append(f"{v}")
    
    plt.yticks(tick_values, tick_labels)
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q Power Spectrum')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(freqs.min(), freqs.max())  # Explicitly set y limits to match actual frequencies
    plt.tight_layout()
    
    plt.savefig('cqt_spectrogram_hz.png')
    plt.show()



