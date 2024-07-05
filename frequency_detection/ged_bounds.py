# %%
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.signal import find_peaks, firwin, lfilter
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import filterFGx, get_base_dir, get_cmap, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()

# %%
# Load the EEG data, leadfield, and channel locations
mat = loadmat('emptyEEG.mat')['EEG']
srate = mat['srate'][0, 0][0, 0]
n_samples = mat['pnts'][0, 0][0, 0]
n_chs = mat['nbchan'][0, 0][0, 0]
lf_gain = mat['lf'][0, 0]['Gain'][0, 0]
ch_names = [ch[0][0] for ch in mat['chanlocs'][0, 0][0]]

# Define simulation parameters
dip_freqs = [[4, 7], [9, 11], [11, 13]]  # Dipole frequency ranges
dip_loc = [1349, 93, 204]  # Dipole location indices

# %%
# Create narrowband nonstationary time series
f_ord = round(30 * (srate / 3))
hz = np.linspace(0, srate, n_samples)
f_kerns = np.zeros((len(dip_loc), f_ord))

# White noise
dip_data = np.random.randn(lf_gain.shape[2], n_samples) * 3

# Pink noise --> optional but unnecessary
# ed = 2000
# for di in tqdm(range(lf_gain.shape[2])):
#     as_ = np.random.rand(n_samples) * np.exp(-(np.arange(n_samples)) / ed)
#     data = np.fft.ifft(as_ * np.exp(1j * 2 * np.pi * np.random.rand(n_samples)))
#     data = np.real(data)
#     dip_data[di, :] = (data - np.mean(data)) / np.std(data)

# Filtering and adding to dipole time series
for i, loc in enumerate(dip_loc):
    # FIR filter
    f_kerns[i, :] = firwin(f_ord, dip_freqs[i], fs=srate, pass_zero=False)
    filt_data = lfilter(f_kerns[i, :], 1, np.random.randn(n_samples)) * 200
    dip_data[loc, :] = filt_data

data = lf_gain[:, 0, :] @ dip_data

# %%
# Plot the dipole time series and the spectrum
fig, axs = plt.subplots(3, 1)
for i, loc in enumerate(dip_loc):
    axs[i].plot(
        [0, dip_freqs[i][0], *dip_freqs[i], dip_freqs[i][1], srate / 2],
        [0, 0, 1, 1, 0, 0],
        'r',
    )
    axs[i].plot(
        np.linspace(0, srate, len(f_kerns[i, :])),
        abs(np.fft.fft(f_kerns[i, :])),
        'ko-',
        mfc='none',
        lw=1,
    )
    sigX = np.abs(np.fft.fft(dip_data[loc, :]))
    axs[i].plot(hz, sigX / max(sigX), lw=1)
    axs[i].set_xlim([0, 20])
    axs[i].set_ylim([0, 1])
    if i == len(axs) - 1:
        axs[i].set_xlabel('Frequency (Hz)')
    axs[i].set_ylabel('Amplitude')
    axs[i].set_title(f'Spectrum, Dipole {i+1}')

plt.tight_layout()
plt.show()

# %%
# Analysis parameters
n_freqs = 80
freqs = np.logspace(np.log10(2), np.log10(30), n_freqs)
stds = np.ones(n_freqs)
onsets = np.arange(2 * srate, n_samples - 2 * srate + 2 * srate, 2 * srate)
n_snip = 2 * srate

# Initialize matrices for GED
evals = np.zeros((n_freqs, n_chs))
evecs = np.zeros((n_freqs, n_chs))
maps = np.zeros((n_freqs, n_chs))

R = np.zeros((n_chs, n_chs))

for onset in onsets:
    snip_data = data[:, onset : onset + n_snip]
    snip_data = snip_data - np.mean(snip_data, axis=1, keepdims=True)
    R += snip_data @ snip_data.T / n_snip
R /= len(onsets)

# Regularize R
gamma = 0.01
Rr = R * (1 - gamma) + np.eye(n_chs) * gamma * np.mean(eigh(R)[0])

# %%
# Filtering and computing matrices
for i in tqdm(range(len(freqs))):
    f_data = filterFGx(data, srate, freqs[i], stds[i])[0]

    # Compute S
    S = np.zeros((n_chs, n_chs))
    for onset in onsets:
        snip_data = f_data[:, onset : onset + n_snip]
        snip_data = snip_data - np.mean(snip_data, axis=1, keepdims=True)
        S += snip_data @ snip_data.T / n_snip

    # Global variance normalization (optional, this scales the eigenspectrum)
    S /= np.std(S) / np.std(R)

    # GED computation
    # Compute the generalized eigenvalues and eigenvectors
    L, W = eigh(S, Rr)

    # Sort the eigenvalues in descending order
    sidx = np.argsort(L)[::-1]
    evals[i, :] = L[sidx]
    W = W[:, sidx]

    # Store top component map and eigenvector
    maps[i, :] = W[:, 0] @ S
    evecs[i, :] = W[:, 0]

# %%
# Calculate the correlation matrix for clustering
scaler = StandardScaler()
E = scaler.fit_transform(evecs.T).T
evec_corr = (E @ E.T / (n_chs - 1)) ** 2

# %%
# Plot the correlation matrix
plt.figure()
plt.contourf(freqs, freqs, 1 - evec_corr, 40, cmap='bone')
plt.xscale('log')
plt.yscale('log')

ticks = np.round(np.logspace(np.log10(1), np.log10(n_freqs), 14), 1)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.axis('square')
plt.title('Eigenvectors Correlation Matrix')

# Add bounding boxes (ground truth)
for tbnds in dip_freqs:
    tbnds_idx = np.searchsorted(freqs, tbnds)
    plt.plot(tbnds, [tbnds[0], tbnds[0]], 'r--', linewidth=2)
    plt.plot(tbnds, [tbnds[1], tbnds[1]], 'r--', linewidth=2)
    plt.plot([tbnds[0], tbnds[0]], tbnds, 'r--', linewidth=2)
    plt.plot([tbnds[1], tbnds[1]], tbnds, 'r--', linewidth=2)

e = evals[:, 0]
e = (e - np.min(e)) / np.max(e)
plt.plot(freqs, 1.5 * e + freqs[0], 'b', linewidth=2)

h = plt.colorbar()
h.set_ticks(np.arange(0, 1.2, 0.2))
h.set_ticklabels(np.arange(0, 1.2, 0.2).round(1)[::-1])
plt.gca().tick_params(labelsize=15)

# Determine the optimal epsilon value
n_epsis = 50
epsis = np.linspace(0.0001, 0.05, n_epsis)
q_vec = np.zeros(n_epsis)

for i in range(n_epsis):
    dbscan = DBSCAN(eps=epsis[i], min_samples=3, metric='correlation')
    dbscan.fit(evec_corr)
    freq_bands = dbscan.labels_

    q_tmp = []
    for i in range(np.max(freq_bands) + 1):
        mask = freq_bands == i
        q_tmp.append(
            np.mean(evec_corr[mask][:, mask]) / np.mean(evec_corr[~mask][:, ~mask])
        )

    q_vec[i] = np.mean(q_tmp) + np.log(np.mean(freq_bands != -1))

peaks, _ = find_peaks(q_vec)
if len(peaks) == 0:
    epsi_idx = n_epsis // 2
else:
    epsi_idx = peaks[np.argmax(q_vec[peaks])]
best_eps = epsis[epsi_idx]

dbscan = DBSCAN(eps=best_eps, min_samples=3, metric='correlation')
dbscan.fit(evec_corr)
freq_bands = dbscan.labels_

# Draw empirical bounds on correlation map
for i in range(np.max(freq_bands) + 1):
    tbnds = freqs[freq_bands == i]
    tbnds = [tbnds[0], tbnds[-1]]

    plt.plot(tbnds, [tbnds[0], tbnds[0]], 'm', linewidth=2)
    plt.plot(tbnds, [tbnds[1], tbnds[1]], 'm', linewidth=2)
    plt.plot([tbnds[0], tbnds[0]], tbnds, 'm', linewidth=2)
    plt.plot([tbnds[1], tbnds[1]], tbnds, 'm', linewidth=2)

# %%
info = create_info(ch_names, srate, 'eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

# Plot the average maps
fig, axs = plt.subplots(3, 3)
axs = axs.ravel()

for i in range(3):

    # Ground truth
    ground_truth = -lf_gain[:, 0, dip_loc[i]]
    plot_topomap(
        ground_truth,
        info,
        axes=axs[i],
        cmap=cmap,
        contours=0,
        show=False,
    )
    axs[i].set_title(f'GT: {np.mean(dip_freqs[i]):.2f} Hz')

    # Maps
    pca_model = PCA().fit(maps[freq_bands == i])
    m = pca_model.components_[0] * np.sign(
        np.corrcoef(pca_model.components_[0], ground_truth)[0, 1]
    )
    plot_topomap(
        m,
        info,
        axes=axs[i + 3],
        cmap=cmap,
        contours=0,
        show=False,
    )
    axs[i + 3].set_title(f'Maps: {np.mean(freqs[freq_bands == i]):.2f} Hz')

    # Eigenvectors
    pca_model = PCA().fit(evecs[freq_bands == i])
    m = pca_model.components_[0] * np.sign(
        np.corrcoef(pca_model.components_[0], ground_truth)[0, 1]
    )
    plot_topomap(
        m,
        info,
        axes=axs[i + 6],
        cmap=cmap,
        contours=0,
        show=False,
    )
    axs[i + 6].set_title(f'E-vecs: {np.mean(freqs[freq_bands == i]):.2f} Hz')

plt.tight_layout()
plt.show()

# %%
