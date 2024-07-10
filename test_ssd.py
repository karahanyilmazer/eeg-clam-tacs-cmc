# %%
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
from os.path import join

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import SSD
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
freqs_sig = 10, 14
freqs_noise = 8, 16

ssd = SSD(
    info=raw.info,
    reg="oas",
    filt_params_signal=dict(
        l_freq=freqs_sig[0],
        h_freq=freqs_sig[1],
        l_trans_bandwidth=1,
        h_trans_bandwidth=1,
    ),
    filt_params_noise=dict(
        l_freq=freqs_noise[0],
        h_freq=freqs_noise[1],
        l_trans_bandwidth=1,
        h_trans_bandwidth=1,
    ),
)
ssd.fit(X=raw.get_data())
ssd.transform(X=raw.get_data())

# %%
# Let's investigate spatial filter with max power ratio.
# We will first inspect the topographies.
# According to Nikulin et al. 2011 this is done by either inverting the filters
# (W^{-1}) or by multiplying the noise cov with the filters Eq. (22) (C_n W)^t.
# We rely on the inversion approach here.

pattern = mne.EvokedArray(data=ssd.patterns_[:4].T, info=ssd.info)
pattern.plot_topomap(units=dict(mag="A.U."), time_format="")

# The topographies suggest that we picked up a parietal alpha generator.

# Transform
ssd_sources = ssd.transform(X=raw.get_data())

# Get psd of SSD-filtered signals.
psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=raw.info["sfreq"], n_fft=4096
)

# Get spec_ratio information (already sorted).
# Note that this is not necessary if sort_by_spectral_ratio=True (default).
spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)

# Plot spectral ratio (see Eq. 24 in Nikulin 2011).
fig, ax = plt.subplots(1)
ax.plot(spec_ratio, color="black")
ax.plot(spec_ratio[sorter], color="orange", label="sorted eigenvalues")
ax.set_xlabel("Eigenvalue Index")
ax.set_ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
ax.legend()
ax.axhline(1, linestyle="--")
# %%
