import matplotlib.pyplot as plt
from typing import Optional


class PlotManager(object):

    @staticmethod
    def plot_thd(ax_thd, plot_x, plot_thd, octave_smoothing: Optional[int] = None):
        """
        Plot THD vs Frequency.

        Args:
            ax_thd: matplotlib axes object
            plot_x: frequency array (Hz)
            plot_thd: THD values array
            octave_smoothing: Optional octave fraction for smoothing (None, 1, 3, 6, 12, 24, 48)
                             None: No smoothing (default)
                             6: 1/6 octave smoothing (recommended for chirp signals)
        """
        # Apply octave smoothing if requested
        if octave_smoothing is not None:
            from base.utils.octave_smoothing import smooth_to_octave_grid
            plot_x, plot_thd = smooth_to_octave_grid(
                plot_x, plot_thd,
                fraction=octave_smoothing,
                method="log"
            )

        ax_thd.plot(plot_x, plot_thd, linewidth=2, alpha=0.7, color=(63 / 255, 160 / 255, 192 / 255), label='THD')
        ax_thd.set_xscale('log')
        ax_thd.set_xlabel("Frequency (Hz)", fontsize=14)
        ax_thd.set_ylabel("THD", fontsize=14)
        ax_thd.set_title("Total Harmonic Distortion (THD) vs Frequency", fontsize=16)
        ax_thd.grid(which='both', linestyle='--', linewidth=0.5)
        ax_thd.legend()
        ax_thd.set_ylim(bottom=0)

    @staticmethod
    def plot_harmonic(ax_harmonic, plot_x, plot_h):
        for i in range(6):
            ax = ax_harmonic[i // 3, i % 3]
            ax.plot(plot_x, plot_h[i], linewidth=1.5, alpha=0.7, label=f'Harmonic_{i}')
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"Harmonic {i} vs Frequency", fontsize=14)
            ax.grid(which='both', linestyle='--', linewidth=0.5)
            ax.legend()
        plt.tight_layout()

    @staticmethod
    def plot_frequency_response(ax_fr, frequency_list, fr):
        ax_fr.plot(frequency_list, fr, linewidth=1.5, alpha=0.7, color="green", label='Frequency Response')
        ax_fr.set_xscale('log')
        ax_fr.set_xlabel("Frequency (Hz)", fontsize=14)
        ax_fr.set_ylabel("Frequency Response (dB)", fontsize=14)
        ax_fr.set_title("The Frequency Response (dB) vs Frequency", fontsize=16)
        ax_fr.grid(which='both', linestyle='--', linewidth=0.5)
        ax_fr.legend()
        ax_fr.set_ylim(bottom=min(fr) - 5)
