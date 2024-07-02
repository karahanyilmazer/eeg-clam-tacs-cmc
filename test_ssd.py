# %%
from os.path import join

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.time_frequency import psd_array_welch
from mne.viz import plot_topomap
from scipy import linalg
from scipy.io import loadmat


# %%
def get_P_TARGET(raw, l_freq, h_freq):
    data_signal = (
        raw.copy()
        .filter(l_freq, h_freq, l_trans_bandwidth=1, h_trans_bandwidth=1)
        ._data
    )  # alpha
    data_noise = (
        raw.copy()
        .filter(8, 16, l_trans_bandwidth=1, h_trans_bandwidth=1)
        .filter(14, 10, l_trans_bandwidth=1, h_trans_bandwidth=1)
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
        plt.semilogy(freqs, psd)
        plt.title(f'{ix_comp:d}: PF = {peak_freq:.2f} Hz')
        plot_topomap(M[:, ix_comp], raw.info)
        plt.show()
    return M[:, 3], peak_freq


# %%
# Read in the data
data_folder = r'C:\Files\Coding\Python\Neuro\eeg-clam-tacs-cmc\data\FS_10'
raw = mne.io.read_raw_brainvision(join(data_folder, 'Relax.vhdr'), preload=True)
raw.drop_channels(['envelope', 'envelope_am', 'force'])
raw.set_montage(make_standard_montage('easycap-M1'), match_case=False)

# Remove the bad channels
bad_idx = loadmat(join(data_folder, 'exclude_idx.mat'))['exclude_idx'].squeeze() - 1
bad_chs = [raw.ch_names[idx] for idx in bad_idx]
raw.drop_channels(bad_chs)

# forward_model = loadmat(join(data_folder, 'P_TARGET_64.mat'))['P_TARGET_64'].squeeze()
# mask_bad = forward_model == 0
# mask_good = ~mask_bad
# bads = np.array(raw.ch_names)[:64][mask_bad]
# forwardModel = forward_model[mask_good]
# raw.drop_channels(bads)

# %%
df = 2
l_freq, h_freq = 10, 14
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
