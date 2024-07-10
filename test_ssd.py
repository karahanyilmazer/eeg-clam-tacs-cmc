# %%
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
from os.path import join

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import psd_array_welch
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import inset_locator
from scipy import linalg

from utils import get_base_dir, get_cmap, read_raw, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()


# %%
def get_P_TARGET(raw, l_freq, h_freq, df, n_comps=4, save=False):
    data_signal = (
        raw.copy()
        .filter(l_freq, h_freq, l_trans_bandwidth=1, h_trans_bandwidth=1)
        ._data
    )
    data_noise = (
        raw.copy()
        .filter(l_freq - df, h_freq + df, l_trans_bandwidth=1, h_trans_bandwidth=1)
        .filter(h_freq, l_freq, l_trans_bandwidth=1, h_trans_bandwidth=1)
        ._data
    )
    data_broad = (
        raw.copy().filter(1, 30, l_trans_bandwidth=1, h_trans_bandwidth=1)._data
    )

    A = np.cov(data_signal)
    B = np.cov(data_noise)
    evals, evecs = linalg.eig(A, B)
    ix = np.argsort(evals)[::-1]
    D = evecs[:, ix].T
    M = linalg.pinv(D)

    if n_comps == 'all':
        n_comps = M.shape[0]

    for ix_comp in range(n_comps):
        psd, freqs = psd_array_welch(
            D[ix_comp] @ data_broad,
            raw.info['sfreq'],
            fmin=1,
            fmax=30,
            n_fft=int(3 * raw.info['sfreq']),
        )
        freq_mask = np.logical_and(freqs > l_freq, freqs < h_freq)
        peak_freq = freqs[freq_mask][np.argmax(psd[freq_mask])]

        fig, ax = plt.subplots()
        ax.axvline(peak_freq, color='tomato', ls='--', lw=0.5)
        ax.semilogy(freqs, psd)

        axins = inset_locator.inset_axes(
            ax, width='30%', height='30%', loc='upper right'
        )
        plot_topomap(M[:, ix_comp], raw.info, axes=axins)

        ax.set_title(f'Component {ix_comp:d}', loc='left')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (dB/Hz)')

        text_str = '\n'.join(
            (
                f'$f_S={l_freq}-{h_freq}$ Hz',
                f'$f_N={l_freq-df}-{h_freq+df}$ Hz',
            )
        )
        props = dict(facecolor='none', edgecolor='black')

        ax_pos = ax.get_position()
        fig.text(
            ax_pos.x1 - 0.007,
            1 - ax_pos.y0,
            text_str,
            horizontalalignment='right',
            verticalalignment='bottom',
            bbox=props,
            transform=fig.transFigure,
        )
        ax.text(
            peak_freq + 0.2,
            0.94,
            f'$PF={peak_freq:.2f}$ Hz',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.get_xaxis_transform(),
            color='tomato',
        )

        folder_path = join(os.path.dirname(__file__), 'img', f'{l_freq}-{h_freq}Hz')
        os.makedirs(folder_path, exist_ok=True)

        if save:
            plt.savefig(
                join(
                    folder_path,
                    f'ssd-{raw.info["subject_info"]["his_id"]}-comp_{ix_comp}.png',
                ),
                dpi=300,
            )
            plt.close()

        else:
            plt.show()

    return M, D, peak_freq


# %%
for subj in os.listdir('data'):
    if int(subj.split('_')[1]) >= 9:
        # Read in the data
        raw = read_raw(subj)

        # Define the SSD parameters
        l_freq, h_freq = 10, 14
        df = 2
        M, D, pf = get_P_TARGET(raw, l_freq, h_freq, df, n_comps='all', save=True)

# %%
x_s = raw.copy().filter(l_freq, h_freq, l_trans_bandwidth=1, h_trans_bandwidth=1)._data
x_n = (
    raw.copy()
    .filter(l_freq - df, h_freq + df, l_trans_bandwidth=1, h_trans_bandwidth=1)
    .filter(h_freq, l_freq, l_trans_bandwidth=1, h_trans_bandwidth=1)
    ._data
)

C_s = (x_s @ x_s.T) / x_s.shape[1]
C_n = (x_n @ x_n.T) / x_n.shape[1]

eig_vals, eig_vecs = linalg.eig(C_s, C_n)

M, pf = get_P_TARGET(raw, l_freq, h_freq)

# %%

# %%
