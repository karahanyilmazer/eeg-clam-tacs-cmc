# %%
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pyvista as pv
import scienceplots
from matplotlib.pyplot import close as close_plt
from mne import create_info
from mne.channels import make_standard_montage
from mne.datasets import fetch_fsaverage
from mne.io import RawArray
from mne.preprocessing import ICA
from mne.viz import plot_topomap
from numpy.fft import fft, ifft
from scipy.io import loadmat
from scipy.linalg import eigh, pinv, toeplitz
from scipy.signal import butter, detrend, filtfilt, hilbert
from sklearn.decomposition import FastICA
from tqdm import tqdm
from yaml import safe_load

from utils import filterFGx, get_base_dir, get_cmap, read_raw, set_style

# Set figure and path settings
base_dir, cmap, _ = get_base_dir(), get_cmap('parula'), set_style()
mne.set_log_level('INFO')
# Set Times New Roman as the global font
plt.rcParams['font.family'] = 'Times New Roman'
plt.style.use(['science', 'no-latex'])


# %%
def compute_plv(signal1, signal2):
    # Compute the analytic signal using the Hilbert transform
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Extract the instantaneous phase
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    # Compute the phase difference
    phase_diff = phase1 - phase2

    # Compute the Phase Locking Value (PLV)
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv


def ssd_orig(X, l_freq, h_freq, df, reduce=True, log=True):
    # Creating filters
    b, a = butter(
        filter_order,
        np.array([l_freq, h_freq]) / (srate / 2),
        btype='bandpass',
    )
    b_f, a_f = butter(
        filter_order,
        np.array([l_freq - df, h_freq + df]) / (srate / 2),
        btype='bandpass',
    )
    b_s, a_s = butter(
        filter_order,
        np.array([l_freq - 1, h_freq + 1]) / (srate / 2),
        btype='bandstop',
    )

    # Covariance matrix for the center frequencies (signal)
    X_s = filtfilt(b, a, X, axis=1)
    C_s = np.cov(X_s)

    # Covariance matrix for the flanking frequencies (noise)
    X_n = filtfilt(b_f, a_f, X, axis=1)
    X_n = filtfilt(b_s, a_s, X_n, axis=1)
    C_n = np.cov(X_n)
    del X_n

    # Eigen decomposition of C
    D, V = eigh(C_s)

    # Sort eigenvalues in descending order and sort eigenvectors accordingly
    # Indices for sorting eigenvalues in descending order
    sort_idx = np.argsort(D)[::-1]
    ev_sorted = D[sort_idx]  # Sorted eigenvalues
    V = V[:, sort_idx]  # Sorted eigenvectors

    # Estimate the rank of the data
    tol = ev_sorted[0] * 10**-6
    r = np.sum(ev_sorted > tol)

    if r < X_s.shape[0] and reduce:
        if log:
            print(
                f'SSD: Input data does not have full rank. Only {r} components can be computed.'
            )
        M = V[:, :r] @ np.diag(ev_sorted[:r] ** -0.5)
    else:
        M = np.eye(X_s.shape[0])

    # Compute reduced covariance matrices
    C_s_r = M.T @ C_s @ M
    C_n_r = M.T @ C_n @ M

    # Solve the generalized eigenvalue problem
    D, W = eigh(C_s_r, C_s_r + C_n_r)

    # Sort eigenvalues and eigenvectors in descending order
    sort_idx = np.argsort(D)[::-1]
    # lambda_sorted = D[sort_idx]
    W = W[:, sort_idx]

    # Compute final matrix W
    W = M @ W

    # Compute matrix A with patterns in columns
    A = C_s @ W @ np.linalg.inv(W.T @ C_s @ W)

    # Apply SSD filters to the data if needed (assuming we want to compute it)
    X_ssd = W.T @ X_s

    return X_ssd, A


# Read in configuration values
with open('config.yaml', 'r') as file:
    config = safe_load(file)

# Read in the raw data
subj = config['excluded_participants'][0]
raw = read_raw(subj)
print(subj)

# Get the number of time points and the sampling rate
srate = int(raw.info['sfreq'] / 20)  # Downsample to 250 Hz
raw.resample(srate, npad="auto")
raw_data, t = raw.get_data(return_times=True)
n_pnts = len(t)

# %% Compute the leadfield matrix
# Create a standard info object with channel names from the montage
montage = make_standard_montage('standard_1020')
info_sim = mne.create_info(ch_names=montage.ch_names, sfreq=srate, ch_types='eeg')
info_sim.set_montage(montage)
# Channels to drop
channels_to_drop = ['T7', 'T8', 'P7', 'P8', 'T3', 'T5', 'T4', 'T6']

# Drop channels from the info object by picking the remaining ones
remaining_channels = [ch for ch in info_sim['ch_names'] if ch not in channels_to_drop]
info_sim = mne.pick_info(
    info_sim, mne.pick_channels(info_sim['ch_names'], remaining_channels)
)
# Fetch the fsaverage dataset
fs_dir = fetch_fsaverage(verbose=True)
trans = 'fsaverage'
subject = ''

# Set up source space for fsaverage
src = mne.setup_source_space(
    subject=subject,
    spacing='oct5',
    subjects_dir=fs_dir,
    add_dist=False,
)

# Load the BEM model for fsaverage
bem = mne.read_bem_solution(fs_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif')

# Compute the forward solution (leadfield matrix)
fwd = mne.make_forward_solution(
    info_sim,
    trans=trans,
    src=src,
    bem=bem,
    meg=False,
    eeg=True,
    mindist=5.0,
    n_jobs=-1,
)

# Convert the forward solution to fixed orientation
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)

# Only pick the channels that are present in the raw data
# fwd = mne.pick_channels_forward(fwd, include=raw.ch_names)

# # Extract the leadfield matrix
lf = fwd['sol']['data']
print('Leadfield matrix shape:', lf.shape)  # (n_channels, n_dipoles)

assert lf.shape[1] == fwd['src'][0]['nuse'] * 2, 'Number of dipoles do not match'

# %%
# Get the index of C3
c3_idx = fwd['info']['ch_names'].index('C3')

# Define the dipole locations
dip_loc1, dip_loc2 = 248, 323
dip_freq1, dip_freq2 = 11, 22
fwhm_filt = 2
fwhm_anal = 5

# Simulate the data
dipole_data = 1 * np.random.randn(lf.shape[1], srate)
# Add signal to the second half of the dataset
dipole_data[dip_loc1, srate // 2 :] = 15 * np.sin(
    2 * np.pi * 11 * np.arange(srate // 2) / srate
)
dipole_data[dip_loc2, srate // 2 :] = 10 * np.sin(
    2 * np.pi * 22 * np.arange(srate // 2) / srate
)
# Project dipole data to scalp electrodes
data = lf @ dipole_data
times = np.squeeze(np.arange(data.shape[1]) / srate)

if True:
    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1:])

    # Mu topoplot
    plot_topomap(lf[:, dip_loc1], info_sim, axes=ax1, show=False, cmap=cmap)
    ax1.plot(-4.75e-02, 1.7e-2, 'o', color='red', markersize=7)
    ax1.set_title(f'Topographic Activation Map', pad=10)
    ax1.set_ylabel('Mu', fontweight='bold')

    # Mu dipole data
    ax2.plot(
        times,
        dipole_data[dip_loc1, :] / np.linalg.norm(dipole_data[dip_loc1, :]),
        linewidth=4,
        label='Dipole',
    )
    ax2.plot(
        times,
        data[c3_idx, :] / np.linalg.norm(data[c3_idx, :]),
        linewidth=2,
        label='Electrode',
    )
    ax2.set_title('Generated Source Activity', pad=10)
    # ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (a.u.)')
    ax2.set_xticklabels([])
    ax2.legend()

    # Beta topoplot
    plot_topomap(lf[:, dip_loc2], info_sim, axes=ax3, show=False, cmap=cmap)
    ax3.plot(-4.75e-02, 1.7e-2, 'o', color='red', markersize=7)
    # ax3.set_title(f'Simulated Beta Activity')
    ax3.set_ylabel('Beta', fontweight='bold')

    # Beta dipole data
    ax4.plot(
        times,
        dipole_data[dip_loc2, :] / np.linalg.norm(dipole_data[dip_loc2, :]),
        linewidth=4,
        label='Dipole',
    )
    ax4.plot(
        times,
        data[c3_idx, :] / np.linalg.norm(data[c3_idx, :]),
        linewidth=2,
        label='Electrode',
    )
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude (a.u.)')
    ax4.legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(
        f'img/thesis/source_activity.pdf', format='pdf', dpi=600, bbox_inches='tight'
    )
    plt.savefig(
        f'img/thesis/source_activity.png', format='png', dpi=600, bbox_inches='tight'
    )
    plt.show()

# %%
# Define the filters to evaluate
filters = {'Best Electrode': 0, 'PCA': 1, 'JD': 2, 'GEDb': 3, 'SSD': 4, 'NID': 5}

freqs = np.array([dip_freq1, dip_freq2])
fwhm_filt = 2
fwhm_anal = 5
n_ch = len(info_sim.ch_names)

spat_maps = np.zeros((n_ch, len(freqs), len(filters)))
corr_data = np.zeros((len(freqs), len(filters)))
snrs = np.zeros((len(freqs), len(filters)))
plvs = np.zeros((len(freqs), len(filters)))


# Loop over frequencies
for fi, freq in enumerate(freqs):
    # Create data time series
    amp1 = 10 + 10 * filterFGx(np.random.randn(n_pnts), srate, 3, 10)[0]
    freq_mod1 = detrend(10 * filterFGx(np.random.randn(n_pnts), srate, 3, 10)[0])
    k1 = (dip_freq1 / srate) * 2 * np.pi / dip_freq1
    signal1 = amp1 * np.sin(2 * np.pi * dip_freq1 * t + k1 * np.cumsum(freq_mod1))

    amp2 = 10 + 10 * filterFGx(np.random.randn(n_pnts), srate, 3, 10)[0]
    freq_mod2 = detrend(10 * filterFGx(np.random.randn(n_pnts), srate, 3, 10)[0])
    k2 = (dip_freq2 / srate) * 2 * np.pi / dip_freq2
    signal2 = amp2 * np.sin(2 * np.pi * dip_freq2 * t + k2 * np.cumsum(freq_mod2))

    # Create dipole data
    pwr_spec = filterFGx(
        (
            np.random.rand(n_pnts, lf.shape[1])
            * np.linspace(-1, 1, n_pnts).reshape(-1, 1) ** 20
        ).T,
        srate,
        10,
        50,
    )[0].T
    data = 100 * np.real(
        ifft(
            pwr_spec + 1j * (2 * np.pi * np.random.rand(*pwr_spec.shape) - np.pi),
            axis=0,
        )
    )

    data[:, dip_loc1] = signal1 + np.random.randn(n_pnts)
    data[:, dip_loc2] = signal2 + np.random.randn(n_pnts)

    # Simulated EEG data
    eeg_data = lf @ data.T
    n_pnts = eeg_data.shape[1]
    eeg_times = times

    if fi == 0:
        signal = signal1
    else:
        signal = signal2

    # Extract data for covariance matrix
    filt_data = filterFGx(eeg_data, srate, freq, fwhm_filt)[0]
    filt_cov = (filt_data @ filt_data.T) / filt_data.shape[1]
    bb_cov = (eeg_data @ eeg_data.T) / n_pnts

    # Find frequency indices
    hz = np.linspace(0, srate, len(t))
    freq_idx = np.searchsorted(hz, freq)
    f_low = np.arange(np.searchsorted(hz, freq - 5), np.searchsorted(hz, freq - 1) + 1)
    f_high = np.arange(np.searchsorted(hz, freq + 1), np.searchsorted(hz, freq + 5) + 1)

    # Best-electrode
    # ==============================================================================
    print('Best Electrode')
    filt_num = filters['Best Electrode']

    el_pwr = np.abs(hilbert(filt_data, axis=0)) ** 2
    max_el = np.argmax(np.mean(el_pwr, axis=1))

    spat_maps[:, fi, filt_num] = np.mean(el_pwr, axis=1)

    tc_data = filterFGx(eeg_data[max_el, :], srate, freq, fwhm_anal)[0]
    corr_data[fi, filt_num] = np.corrcoef(tc_data, signal)[0, 1] ** 2

    plvs[fi, filt_num] = compute_plv(tc_data, signal)

    f = np.abs(fft(eeg_data[max_el, :]) / n_pnts) ** 2
    snrs[fi, filt_num] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

    # PCA
    # ==============================================================================
    print('PCA')
    filt_num = filters['PCA']

    evals, pca_vecs = eigh(filt_cov)
    pca_data = (filterFGx(eeg_data, srate, freq, fwhm_anal)[0].T @ pca_vecs).T
    fft_pwr = np.abs(fft(pca_data, axis=1)) ** 2
    best_comp = np.argmax(fft_pwr[:, np.searchsorted(hz, freq)])
    maps = pca_vecs[:, best_comp]

    idx = np.argmax(np.abs(maps))
    spat_maps[:, fi, filt_num] = maps * np.sign(maps[idx])

    corr_data[fi, filt_num] = np.corrcoef(pca_data[best_comp, :], signal)[0, 1] ** 2

    plvs[fi, filt_num] = compute_plv(pca_data[best_comp, :], signal)

    pca_data = eeg_data.T @ pca_vecs[:, idx]
    f = np.abs(fft(pca_data) / n_pnts) ** 2
    snrs[fi, filt_num] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

    # GEDb
    # ==============================================================================
    filt_num = filters['GEDb']
    print('GED')

    evals, ged_vecs = eigh(filt_cov, bb_cov)

    max_idx = np.argmax(evals)
    maps = filt_cov @ ged_vecs @ np.linalg.inv(ged_vecs.T @ filt_cov @ ged_vecs)
    maps = maps[:, max_idx]

    ged_data = filterFGx(eeg_data, srate, freq, fwhm_anal)[0].T @ ged_vecs[:, max_idx]

    idx = np.argmax(np.abs(maps))
    spat_maps[:, fi, filt_num] = maps * np.sign(maps[idx])

    corr_data[fi, filt_num] = np.corrcoef(ged_data, signal)[0, 1] ** 2

    plvs[fi, filt_num] = compute_plv(ged_data, signal)

    ged_data = eeg_data.T @ ged_vecs[:, max_idx]
    f = np.abs(fft(ged_data) / n_pnts) ** 2
    snrs[fi, filt_num] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

    # SSD
    # ==============================================================================
    filt_num = filters['SSD']
    print('SSD')

    df = 2
    l_freq = freq - df
    h_freq = freq + df
    filter_order = 2

    X_ssd, A = ssd_orig(eeg_data, l_freq, h_freq, df)

    idx = np.argmax(np.abs(A[:, 0]))
    spat_maps[:, fi, filt_num] = A[:, 0] * np.sign(A[idx, 0])
    corr_data[fi, filt_num] = np.corrcoef(X_ssd[0, :], signal)[0, 1] ** 2
    plvs[fi, filt_num] = compute_plv(X_ssd[0, :], signal)
    f = np.abs(fft(X_ssd[0, :]) / n_pnts) ** 2
    snrs[fi, filt_num] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

# %%
# NID
# ==============================================================================
filt_num = filters['NID']
print('NID')

df = 2
filter_order = 2
f_m = 1
f_n = 2

X_ssd1, A1 = ssd_orig(eeg_data, dip_freq1 - df, dip_freq1 + df, df, reduce=True)
X_ssd2, A2 = ssd_orig(eeg_data, dip_freq2 - 2, dip_freq2 + 2, df, reduce=True)

X_stacked = np.vstack([X_ssd1, X_ssd2])

# Run ICA
channel_names = [f'EEG {i}' for i in range(X_stacked.shape[0])]  # Channel names
channel_types = ['eeg'] * X_stacked.shape[0]  # Assume all are EEG channels
info_ica = mne.create_info(ch_names=channel_names, sfreq=srate, ch_types=channel_types)
raw = RawArray(X_stacked, info_ica)
ica = ICA(
    max_iter='auto',
    method='fastica',
    # fit_params=dict(extended=True),
)
print('Fitting ICA')
ica.fit(raw)

A_ica = ica.mixing_matrix_
assert A_ica.shape[0] == A1.shape[1] + A2.shape[1]

A_final1 = A1 @ A_ica[: A1.shape[1], :]
A_final2 = A2 @ A_ica[A1.shape[1] :, :]


# Store the spatial maps
idx = np.argmax(np.abs(A_final1[:, 0]))
spat_maps[:, 0, filt_num] = A_final1[:, 0] * np.sign(A_final1[idx, 0])
corr_data[0, filt_num] = np.corrcoef(X_ssd1[0, :], signal1)[0, 1] ** 2
plvs[0, filt_num] = compute_plv(X_ssd1[0, :], signal1)
f = np.abs(fft(X_ssd1[0, :]) / n_pnts) ** 2
hz = np.linspace(0, srate, len(t))
freq_idx = np.searchsorted(hz, dip_freq1)
f_low = np.arange(
    np.searchsorted(hz, dip_freq1 - 5), np.searchsorted(hz, dip_freq1 - 1) + 1
)
f_high = np.arange(
    np.searchsorted(hz, dip_freq1 + 1), np.searchsorted(hz, dip_freq1 + 5) + 1
)
snrs[0, filt_num] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])


idx = np.argmax(np.abs(A_final2[:, 0]))
spat_maps[:, 1, filt_num] = A_final2[:, 0] * np.sign(A_final2[idx, 0])
corr_data[1, filt_num] = np.corrcoef(X_ssd2[0, :], signal2)[0, 1] ** 2
plvs[1, filt_num] = compute_plv(X_ssd2[0, :], signal2)
f = np.abs(fft(X_ssd2[0, :]) / n_pnts) ** 2
hz = np.linspace(0, srate, len(t))
freq_idx = np.searchsorted(hz, dip_freq2)
f_low = np.arange(
    np.searchsorted(hz, dip_freq2 - 5), np.searchsorted(hz, dip_freq2 - 1) + 1
)
f_high = np.arange(
    np.searchsorted(hz, dip_freq2 + 1), np.searchsorted(hz, dip_freq2 + 5) + 1
)
snrs[1, filt_num] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

# %%
# Find the indices of the frequencies
freqs_to_plot = [np.abs(freqs - f).argmin() for f in freqs]
filts_to_plot = ['Ground Truth', 'Best Electrode', 'PCA', 'GEDb', 'SSD', 'NID']
filts_to_plot = ['Ground Truth', 'PCA', 'GEDb', 'SSD', 'NID']

cmap = get_cmap('parula')
fig, axes = plt.subplots(
    len(freqs_to_plot),
    len(filts_to_plot),
    figsize=(len(freqs_to_plot) * 8, len(filters) * 0.5),
)

for fi in range(len(freqs_to_plot)):
    for i, filt in enumerate(filts_to_plot):
        ax = axes[fi, i]
        if i == 0:
            if fi == 0:
                plot_topomap(lf[:, dip_loc1], info_sim, axes=ax, show=False, cmap=cmap)
                ax.set_title('Ground Truth')
            else:
                plot_topomap(lf[:, dip_loc2], info_sim, axes=ax, show=False, cmap=cmap)

        else:
            # Create the topomap
            plot_topomap(
                np.real(spat_maps[:, freqs_to_plot[fi], filters[filt]]),
                info_sim,
                axes=ax,
                show=False,
                # contours=0,
                cmap=cmap,
            )

            # Set the title
            if fi == 0:
                ax.set_title(filt)
        if i == 0:
            ax.set_ylabel(f'{round(freqs[freqs_to_plot[fi]])} Hz')

plt.savefig(os.path.join('img', 'thesis', 'topo_no_noise.png'))
plt.show()

# %%
filts_to_plot = ['PCA', 'GEDb', 'SSD', 'NID']
fig, axs = plt.subplots(3, 1, figsize=(9, 9))

mean_corr = np.nanmean(corr_data, axis=0)
mean_snrs = np.nanmean(snrs, axis=0)
mean_plvs = np.nanmean(plvs, axis=0)

# Number of bars
n_bars = len(filts_to_plot)

# Create an array for the x-axis
index = np.arange(n_bars)

# Set the bar width
bar_width = 0.6

# Plot each set of bars
for i, filt in enumerate(filts_to_plot):
    filt_num = filters[filt]
    axs[0].bar(index[i], mean_corr[filt_num], bar_width)
    axs[1].bar(index[i], mean_snrs[filt_num], bar_width)
    axs[2].bar(index[i], mean_plvs[filt_num], bar_width)

# Set x-axis labels and legend
axs[0].set_xticks(index)
# axs[0].set_xticklabels(filts_to_plot)
axs[0].set_xticklabels([])
axs[0].set_ylabel('Goodness of Fit ($R^2$)')
axs[0].set_ylim(0, 1)

axs[1].set_xticks(index)
# axs[1].set_xticklabels(filts_to_plot)
axs[1].set_xticklabels([])
axs[1].set_ylabel('SNR')

axs[2].set_xticks(index)
axs[2].set_xticklabels(filts_to_plot)
axs[2].set_ylabel('PLV')
axs[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join('img', 'thesis', 'bar_plot.png'))
plt.show()

# %%
