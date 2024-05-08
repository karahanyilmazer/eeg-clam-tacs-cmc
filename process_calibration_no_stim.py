import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.time_frequency import psd_array_welch
from mne.viz import plot_topomap
from scipy import linalg
from scipy.io import savemat

# from utils import plot_flip


def wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi


def get_P_TARGET(raw):
    raw.set_montage(make_standard_montage('easycap-M1'), match_case=False)
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
        plt.title('{:d}, pf = {:.2f} Hz'.format(ix_comp, peak_freq))
        plot_topomap(M[:, ix_comp], raw.info)
        plt.show()
    return M[:, 3], peak_freq


montage = make_standard_montage('easycap-M1')

# CHANGE HERE BAD ELECTRODES !!!!
bads = [
    'P2',
    'PO4',
    'P4',
    'AF4',
    'F2',
    'F4',
    'O2',
    'C5',
    'O1',
    'F10',
    'TP9',
    'F7',
    'FT10',
    'T8',
]
### CHECK IF AFZ GOOD OR BAD!

l_freq = 10
h_freq = 14

# Load the data without stimulation
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(
    initialdir="C:\\Files\\Coding\\Python\\Neuro\\eeg-clam-tacs-cmc\\data"
)
raw = mne.io.read_raw_brainvision(folder_path + '\\Relax.vhdr', preload=True)
raw.drop_channels(['envelope', 'envelope_am', 'force'])

# Save the indexes to exclude as well as the channels to use for the experiment
exclude_idx = np.sort([raw.ch_names.index(ch) + 1 for ch in bads])
data_dict = {"exclude_idx": exclude_idx}
savemat(folder_path + "/exclude_idx.mat", data_dict)

good_indices = np.where(~np.isin(raw.ch_names, bads))[0]
raw.drop_channels(bads)
# Re-reference the data to parietal electrodes to attenuate occipital alpha components
# raw_ref = raw.copy().set_eeg_reference(ref_channels=['Pz'])

P_TARGET, peak_freq = get_P_TARGET(raw)

raw.filter(l_freq, h_freq)
# flip = plot_flip(raw,P_TARGET)
plt.close()

# Save P_TARGET_64 for matlab
P_TARGET_64 = np.zeros((1, 64))
P_TARGET_64[0, good_indices] = P_TARGET
data_dict = {'P_TARGET_64': P_TARGET_64}
savemat(folder_path + "/P_TARGET_64.mat", data_dict)

# Save phase delay for matlab
data_dict = {'phase_delay': 0.0}
savemat(folder_path + "/phase_delay.mat", data_dict)

# Save flip for matlab
# data_dict = {'flip': flip}
# savemat(folder_path + "/flip.mat", data_dict)
