import matplotlib.pyplot as plt


def plot_thd(ax_thd, plot_x, plot_thd):
    ax_thd.plot(plot_x, plot_thd, linewidth=1, alpha=0.2)
    ax_thd.set_xscale('log')
    ax_thd.set_xlabel("Frequency (Hz)", fontsize=12)
    ax_thd.set_ylabel("THD", fontsize=12)
    ax_thd.set_title("Total Harmonic Distortion (THD) vs Frequency", fontsize=14)



def plot_harmonic(ax_harmonic, plot_x, plot_h):
    for i in range(6):
        ax = ax_harmonic[i // 3, i % 3]
        ax.plot(plot_x, plot_h[i], linewidth=1, alpha=0.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Harmonic_{i} vs Frequency", fontsize=12)
    plt.tight_layout()


def plot_frequency_response(ax_fr, frequency_list, fr):
    ax_fr.plot(frequency_list, fr, linewidth=1, alpha=0.2)
    ax_fr.set_xscale('log')
    ax_fr.set_xlabel("Frequency (Hz)", fontsize=12)
    ax_fr.set_ylabel("Frequency Response (dB)", fontsize=12)
    ax_fr.set_title("The Frequency Response (dB) vs Frequency", fontsize=14)




