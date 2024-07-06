# %%
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from mne.viz import plot_topomap
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import filterFGx, get_base_dir, get_cmap, read_raw, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()

# %%
# Load the EEG data
raw = read_raw('FS_01')
srate = int(raw.info['sfreq'])
data = raw.get_data()
n_chs, n_samples = data.shape
# savemat(
#     'data.mat',
#     {'data': data, 'srate': srate, 'ch_names': raw.ch_names, 'times': raw.times},
# )

# %%
# Analysis parameters
n_freqs = 100
low_freq, high_freq = 2, 50
freqs = np.logspace(np.log10(low_freq), np.log10(high_freq), n_freqs)
stds = np.linspace(2, 5, n_freqs)
onsets = np.arange(2 * srate, n_samples - 4 * srate + 2 * srate, 2 * srate)
n_snip = 2 * srate

# Initialize matrices for GED
evals = np.zeros((n_freqs, n_chs))
evecs = np.zeros((n_freqs, n_chs))
maps = np.zeros((n_freqs, n_chs))

R = np.zeros((len(onsets), n_chs, n_chs))

for i, onset in enumerate(onsets):
    snip_data = data[:, onset : onset + n_snip]
    snip_data = snip_data - np.mean(snip_data, axis=1, keepdims=True)
    R[i, :, :] = snip_data @ snip_data.T / n_snip

#  Compute the Euclidean distance of each segment from the mean
dists = np.zeros(len(onsets))
for i in range(len(onsets)):
    dists[i] = np.linalg.norm(R[i] - np.mean(R, axis=0))
    # dists[i] = np.sqrt(np.sum((R[i] - np.mean(R, axis=0)) ** 2))

# Z-score normalize the distances
z_dists = zscore(dists)

# Select segments with a Z-score less than 3 and compute their mean
R = np.mean(R[z_dists < 3, :, :], axis=0)

# Regularize R
gamma = 0.01
Rr = R * (1 - gamma) + np.eye(n_chs) * gamma * np.mean(eigh(R)[0])

# %%
# Filtering and computing matrices
for i in tqdm(range(len(freqs))):
    f_data = filterFGx(data, srate, freqs[i], stds[i])[0]

    # Compute S
    S = np.zeros((len(onsets), n_chs, n_chs))
    dists = np.zeros(len(onsets))

    for j, onset in enumerate(onsets):
        snip_data = f_data[:, onset : onset + n_snip]
        snip_data = snip_data - np.mean(snip_data, axis=1, keepdims=True)
        S[j, :, :] = snip_data @ snip_data.T / n_snip
        dists[j] = np.linalg.norm(S[i] - np.mean(S, axis=0))

    # Select segments with a Z-score less than 3 and compute their mean
    S = np.mean(S[zscore(dists) < 3, :, :], axis=0)

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

e = evals[:, 0]
e = (e - np.min(e)) / np.max(e)
plt.plot(freqs, 1.5 * e + freqs[0], 'b', linewidth=2)

h = plt.colorbar()
h.set_ticks(np.arange(0, 1.2, 0.2))
h.set_ticklabels(np.arange(0, 1.2, 0.2).round(1)[::-1])
plt.gca().tick_params(labelsize=15)

# Determine the optimal epsilon value
n_epsis = 100
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
# Find unique clusters
unique_clusters = np.unique(freq_bands[freq_bands >= 0])

# Initialize dictionary to hold the start and end frequencies of each cluster
cluster_bounds = {}

for cluster in unique_clusters:
    indices = np.where(freq_bands == cluster)[0]
    start_freq = freqs[indices[0]]
    end_freq = freqs[indices[-1]]
    cluster_bounds[cluster] = (start_freq, end_freq)

# Display the start and end frequencies for each cluster
for cluster, bounds in cluster_bounds.items():
    low, high = bounds[0].round(2), bounds[1].round(2)
    print(f"Cluster {cluster}: Start Frequency = {low}\tEnd Frequency = {high}")

# %%
