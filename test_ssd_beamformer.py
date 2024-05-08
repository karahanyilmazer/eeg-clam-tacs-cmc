"""
To compute SSD on xdf files - from the BeAM BCI
-Possibility to test CCA (cross-covariance)
-Test MNS data

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyxdf

matplotlib.use('Qt5Agg')
import tkinter as tk
from tkinter import filedialog

import mne
from mne.channels import make_standard_montage
from mne.time_frequency import psd_array_welch
from mne.viz import plot_topomap
from scipy import linalg
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA


def get_ssd_forward_model(raw):  # SSD forward model - SNR
    raw.set_montage(make_standard_montage('easycap-M1'), match_case=False)
    data_alpha = (
        raw.copy().filter(10, 14, l_trans_bandwidth=1, h_trans_bandwidth=1)._data
    )
    data_broad = (
        raw.copy().filter(1, 30, l_trans_bandwidth=1, h_trans_bandwidth=1)._data
    )
    data_noise = (
        raw.copy()
        .filter(8, 16, l_trans_bandwidth=1, h_trans_bandwidth=1)
        .filter(14, 10, l_trans_bandwidth=1, h_trans_bandwidth=1)
        ._data
    )
    A = np.cov(data_alpha)  # +np.cov(data_beta)
    B = np.cov(data_noise)
    evals, evecs = linalg.eig(A, B)
    ix = np.argsort(evals)[::-1]
    D = evecs[:, ix].T
    M = linalg.pinv(D)

    for ix_comp in range(5):
        psd, freqs = psd_array_welch(
            D[ix_comp] @ data_broad,
            raw.info['sfreq'],
            fmin=1,
            fmax=30,
            n_fft=int(3 * raw.info['sfreq']),
        )
        freq_mask = np.logical_and(freqs > 8, freqs < 14)
        peak_freq = freqs[freq_mask][np.argmax(psd[freq_mask])]
        # plt.figure()
        plt.semilogy(freqs, psd)
        plt.title('{:d}, pf = {:.2f} Hz'.format(ix_comp, peak_freq))
        # plt.figure()
        plt.title('{:d}, pf = {:.2f} Hz'.format(ix_comp, peak_freq))
        plot_topomap(M[:, ix_comp], raw.info)
        plt.show()
    return M[:, 0]


def plot_forward_model(raw, forward_model):
    raw.set_montage(make_standard_montage('easycap-M1'), match_case=False)
    data_broad = (
        raw.copy().filter(1, 30, l_trans_bandwidth=1, h_trans_bandwidth=1)._data
    )  # ,l_trans_bandwidth=1, h_trans_bandwidth=1
    psd, freqs = psd_array_welch(
        forward_model @ data_broad,
        raw.info['sfreq'],
        fmin=1,
        fmax=30,
        n_fft=int(3 * raw.info['sfreq']),
    )
    freq_mask = np.logical_and(freqs > 8, freqs < 14)
    peak_freq = freqs[freq_mask][np.argmax(psd[freq_mask])]
    plt.semilogy(freqs, psd)
    plt.title(peak_freq)
    # plt.title(peak_freq)
    plot_topomap(forward_model, raw.info)
    plt.show()


def get_ssd_forward_model_old(raw):  # SSD forward model - SNR
    raw.set_montage(make_standard_montage('easycap-M1'), match_case=False)
    data_alpha = raw.copy().filter(10, 14)._data  #
    # data_beta
    data_broad = (
        raw.copy().filter(1, 30)._data
    )  # ,l_trans_bandwidth=1, h_trans_bandwidth=1
    data_noise = (
        raw.copy()
        .filter(1, 30)
        .filter(14, 10, l_trans_bandwidth=1, h_trans_bandwidth=1)
        ._data
    )
    A = np.cov(data_alpha)  # +np.cov(data_beta)
    B = np.cov(data_noise)
    evals, evecs = linalg.eig(A, B)
    ix = np.argsort(evals)[::-1]
    D = evecs[:, ix].T
    M = linalg.pinv(D)

    for ix_comp in range(5):
        psd, freqs = psd_array_welch(
            D[ix_comp] @ data_broad,
            raw.info['sfreq'],
            fmin=1,
            fmax=30,
            n_fft=int(3 * raw.info['sfreq']),
        )
        freq_mask = np.logical_and(freqs > 8, freqs < 14)
        peak_freq = freqs[freq_mask][np.argmax(psd[freq_mask])]
        # plt.figure()
        plt.semilogy(freqs, psd)
        plt.title('{:d}, pf = {:.2f} Hz'.format(ix_comp, peak_freq))
        # plt.figure()
        plt.title('{:d}, pf = {:.2f} Hz'.format(ix_comp, peak_freq))
        plot_topomap(M[:, ix_comp], raw.info)
        plt.show()
    return M[:, 0]


def get_cca_forward_model(raw):
    raw.set_montage('easycap-M1', match_case=False)
    data_alpha = (
        raw.copy().filter(10, 14, l_trans_bandwidth=1, h_trans_bandwidth=1)._data
    )
    data_beta = (
        raw.copy().filter(20, 28, l_trans_bandwidth=1, h_trans_bandwidth=1)._data
    )  # .apply_hilbert(envelope=True).filter(10,14,l_trans_bandwidth=1,h_trans_bandwidth=1)._data
    data_noise = (
        raw.copy()
        .filter(8, 16, l_trans_bandwidth=1, h_trans_bandwidth=1)
        .filter(14, 10, l_trans_bandwidth=1, h_trans_bandwidth=1)
        ._data
    )
    data_broad = raw.copy().filter(1, 30)._data
    n_chs = len(raw.ch_names)
    cca = CCA(n_components=n_chs)
    cca.fit(data_alpha.T, data_beta.T)
    L = cca.x_loadings_
    M = np.diag(np.std(data_alpha, axis=-1)) @ L
    fig, axs = plt.subplots(2, 5, figsize=(12, 4), layout='constrained')
    for ix_comp in range(5):
        # psd,freqs = psd_array_welch(make_target(data_broad,M[:,ix_comp]),raw.info['sfreq'],fmin=1,fmax=30,n_fft=int(3*raw.info['sfreq']))
        psd, freqs = psd_array_welch(
            data_broad @ M[:, ix_comp],
            raw.info['sfreq'],
            fmin=1,
            fmax=30,
            n_fft=int(3 * raw.info['sfreq']),
        )
        freq_mask = np.logical_and(freqs > 1, freqs < 30)
        peak_freq = freqs[freq_mask][np.argmax(psd[freq_mask])]
        axs[1, ix_comp].semilogy(freqs, psd)
        plot_topomap(M[:, ix_comp], raw.info, axes=axs[0, ix_comp], show=False)
        axs[0, ix_comp].set_title('{:d}'.format(ix_comp))
        plt.show()
    return M[:, 0]


def make_target(data, forward_model, COV=None):
    n_chs = len(forward_model)
    if data.ndim == 3:
        data = data[:, :n_chs]
        if COV is None:
            COV = np.cov(np.real(np.concatenate(data, axis=-1)))
        COVinv = linalg.pinv(COV)
        w = ((COVinv @ forward_model[:, None])).squeeze() / (
            forward_model[None, :] @ COVinv @ forward_model[:, None]
        )
        return np.array([w @ ep for ep in data]).squeeze()
    else:
        data = data[:n_chs]
        if COV is None:
            COV = np.cov(np.real(data))
        COVinv = linalg.pinv(COV)
        w = ((COVinv @ forward_model[:, None])).squeeze() / (
            forward_model[None, :] @ COVinv @ forward_model[:, None]
        )
        return (w @ data).squeeze()


def create_mne_raw(stream, marker_stream, beamformer):
    # Extract information from stream
    t_reference = stream['time_stamps']
    sfreq = np.float(stream['info']['effective_srate'])
    ch_names = ['beam']
    data = beamformer.reshape(1, len(beamformer))
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data=data, info=info)
    raw.info['temp'] = t_reference

    markers = np.array(marker_stream['time_series'])[:, 7]
    onsets = (
        marker_stream['time_stamps'] - t_reference[0]
    )  # onset from EEG stream or markers stream?
    annotations = mne.Annotations(onsets, [0] * len(onsets), markers)
    raw.set_annotations(annotations)

    return raw


# CHANGE HERE BAD ELECTRODES !!!!
# default_bads = []
# default_bads = ['AF3','F5','F7','F3','P3','PO3','P5','C6']
# bads = default_bads + []

# file_nostim = 'C:/Users/Annalisa Colucci/Desktop/JupyterNotebook/ClosedLoop/Data/MNS_study/P12NoStimulation_02.xdf'
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(
    initialdir="C:\\Users\\Annalisa Colucci\\Desktop\\\BCI_softwares\\BeamBCI-ClosedLoopStimulation\\Data\\Force Study Taisiia"
)
raw = mne.io.read_raw_brainvision(folder_path + '\\Relax.vhdr', preload=True)
raw.drop_channels(['envelope', 'envelope_am', 'force'])
# Load bad channel names
# simulated_beamformer = loadmat('{}\\eeg_env_simulation.mat'.format(folder_path))['to_store'].squeeze()
forward_model = loadmat('{}\\P_TARGET_64.mat'.format(folder_path))[
    'P_TARGET_64'
].squeeze()
mask_bad = forward_model == 0
mask_good = ~mask_bad
bads = np.array(raw.ch_names)[:64][mask_bad]
forwardModel = forward_model[mask_good]

# Extract bad channels from bad_idx file
# bads_idx = loadmat('{}\\exclude_idx.mat'.format(folder_path))['exclude_idx'].squeeze()
# bads = []
# for ch in bads_idx-1:
#    bads.append(raw.ch_names[ch])

good_indices = np.where(~np.isin(raw.ch_names, bads))[0]
raw.drop_channels(bads)
# raw_ref = raw.copy().set_eeg_reference(ref_channels=['Pz'])


# Try plotting the online selected forward model
plot_forward_model(raw, forwardModel)
forward_model_new = get_ssd_forward_model(raw)


# forward_model_cca = get_cca_forward_model(raw)

# Compute backward model - Beamformer
# Bandpass data into alpha
raw.filter(10, 14)
beamformer = make_target(raw.data, forward_model_new)

print('end')
