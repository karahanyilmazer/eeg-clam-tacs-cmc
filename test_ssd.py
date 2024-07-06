# %%
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
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
def get_P_TARGET(raw, l_freq, h_freq, df):
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

    for ix_comp in range(4):
        psd, freqs = psd_array_welch(
            D[ix_comp] @ data_broad,
            raw.info['sfreq'],
            fmin=1,
            fmax=30,
            n_fft=int(3 * raw.info['sfreq']),
        )
        freq_mask = np.logical_and(freqs > l_freq, freqs < h_freq)
        peak_freq = freqs[freq_mask][np.argmax(psd[freq_mask])]

        _, ax = plt.subplots()
        ax.axvline(peak_freq, color='grey', ls='--', lw=0.5)
        ax.semilogy(freqs, psd)

        axins = inset_locator.inset_axes(
            ax, width='30%', height='30%', loc='upper right'
        )
        plot_topomap(M[:, ix_comp], raw.info, axes=axins)

        ax.set_title(f'Component {ix_comp:d}: PF = {peak_freq:.2f} Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (dB/Hz)')
        plt.show()

    return M[:, 3], peak_freq


# %%
# Read in the data
raw = read_raw('FS_10')

# Define the SSD parameters
l_freq, h_freq = 10, 14
df = 2
M, pf = get_P_TARGET(raw, l_freq, h_freq, df)

# forward_model = loadmat(join(data_folder, 'P_TARGET_64.mat'))['P_TARGET_64'].squeeze()
# mask_bad = forward_model == 0
# mask_good = ~mask_bad
# bads = np.array(raw.ch_names)[:64][mask_bad]
# forwardModel = forward_model[mask_good]
# raw.drop_channels(bads)

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
