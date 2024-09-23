import matplotlib.pyplot as plt


def plot_thd(ax1, plot_x, plot_thd):
    ax1.plot(plot_x, plot_thd, linewidth=1, alpha=0.2)
    ax1.set_xscale('log')
    ax1.set_xlabel("Frequency (Hz)", fontsize=12)
    ax1.set_ylabel("THD", fontsize=12)
    ax1.set_title("Total Harmonic Distortion (THD) vs Frequency", fontsize=14)



def plot_harmonic(ax2, plot_x, plot_h):
    for i in range(6):
        ax = ax2[i // 3, i % 3]
        ax.plot(plot_x, plot_h[i], linewidth=1, alpha=0.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Harmonic_{i} vs Frequency", fontsize=12)
    plt.tight_layout()


def plot_frequency_response(ax3, frequency_list, fr):
    ax3.plot(frequency_list, fr, linewidth=1, alpha=0.2)
    ax3.set_xscale('log')
    ax3.set_xlabel("Frequency (Hz)", fontsize=12)
    ax3.set_ylabel("Frequency Response (dB)", fontsize=12)
    ax3.set_title("The Frequency Response (dB) vs Frequency", fontsize=14)




