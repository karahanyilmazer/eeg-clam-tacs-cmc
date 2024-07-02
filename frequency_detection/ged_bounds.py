# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, firwin, lfilter
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# %%
# Load the EEG data, leadfield, and channel locations
data = loadmat('emptyEEG.mat')
EEG = data['EEG']
srate = EEG['srate'][0, 0][0, 0]
n_samples = EEG['pnts'][0, 0][0, 0]
n_chans = n_chans[0, 0][0, 0]
lf_gain = EEG['lf'][0, 0]['Gain'][0, 0]

# Define frequency boundaries
dipole_freqs = [[4, 7], [9, 11], [11, 13]]

# Dipole locations
dipole_loc = [1350, 94, 205]

# %%
# Create narrowband nonstationary time series
hz = np.linspace(0, srate, n_samples)
dip_data = np.random.randn(lf_gain.shape[2], n_samples) * 3

# Filtering and adding to dipole time series
# for di, loc in enumerate(dipole_loc):
di = 0
loc = dipole_loc[di]
# FIR filter
f_ord = round(30 * (srate / 3))
f_kern = firwin(f_ord, np.array(dipole_freqs[di]) / (srate / 2), pass_zero=False)
filt_data = lfilter(f_kern, 1, np.random.randn(n_samples)) * 200
dip_data[loc, :] = filt_data

# Plot
# plt.figure(1)
# plt.subplot(3, 1, di + 1)
# plt.plot(hz, abs(np.fft.fft(f_kern)), 'ko-')
# plt.plot(
#     [0, dipole_freqs[di][0], dipole_freqs[di][1], srate / 2], [0, 0, 1, 0], 'r'
# )
# plt.xlim([0, 20])
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.title(f'Spectrum, dipole {di+1}')

data = np.squeeze(np.matmul(lf_gain[:, 1, :], dip_data))

# %%
# Analysis parameters
n_freqs = 80
frex = np.logspace(np.log10(2), np.log10(30), n_freqs)
stds = np.linspace(1, 1, n_freqs)
onsets = range(srate * 2, n_samples - srate * 2, 2 * srate)
snipn = 2 * srate

# Initialize matrices for GED
evals = np.zeros((n_freqs, n_chans))
evecs = np.zeros((n_freqs, n_chans))
maps = np.zeros((n_freqs, n_chans))

# Regularization for the covariance matrix
gamma = 0.01

# Filtering and computing matrices
for fi, freq in enumerate(frex):
    fdat = lfilter(*firwin(srate, freq, stds[fi]), EEG['data'])

    # Compute matrices
    S = np.zeros((n_chans, n_chans))
    R = np.zeros_like(S)
    for onset in onsets:
        snipdat = fdat[:, onset : onset + snipn]
        snipdat -= np.mean(snipdat, axis=1, keepdims=True)
        S += np.dot(snipdat, snipdat.T) / snipn
        R += np.dot(snipdat, snipdat.T) / snipn
    R /= len(onsets)
    Rr = R * (1 - gamma) + np.eye(n_chans) * gamma * np.mean(np.linalg.eigvalsh(R))

    # GED computation
    L, W = np.linalg.eigh(S, Rr)
    idx = L.argsort()[::-1]
    evals[fi, :] = L[idx]
    evecs[fi, :] = W[:, idx[0]]
    maps[fi, :] = np.dot(W[:, idx[0]].T, S)

# Continue as in MATLAB for correlation matrices, clustering, etc.
# %%
