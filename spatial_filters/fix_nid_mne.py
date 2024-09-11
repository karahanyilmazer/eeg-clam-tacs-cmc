# %%
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from fastICA import fastICA
# from jade import jadeR
# from sobi import sobi
# from jadeR import jadeR
from utils import filterFGx, get_base_dir, get_cmap

sys.path.insert(0, os.path.join(get_base_dir(), 'eeg-classes'))

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA
from mne.viz import plot_topomap
from numpy.fft import fft, ifft
from scipy.io import loadmat
from scipy.linalg import eigh, pinv, toeplitz
from scipy.signal import butter, detrend, filtfilt, hilbert
from sklearn.decomposition import FastICA

plt.rcParams.update({'figure.dpi': 300})
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
    D, W = eigh(C_s_r, C_n_r)

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


# Load data from the mat file
mat = loadmat('emptyEEG.mat')
EEG = mat['EEG'][0, 0]
lf = mat['lf'][0, 0][2]

orig_EEG = EEG.copy()

# Filter parameters
dip_freq1 = 11
dip_freq2 = 22
freqs = np.array([dip_freq1, dip_freq2])
fwhm_filt = 2
fwhm_anal = 5

# Indices of dipole locations
# dip_loc1 = 93
# dip_loc2 = 204
dip_loc1 = 445
dip_loc2 = 456
orientation = 1  # 0 for "EEG" and 1 for "MEG"

# Define the filters to evaluate
filters = {'Best Electrode': 0, 'PCA': 1, 'JD': 2, 'GEDb': 3, 'SSD': 4, 'NID': 5}

# Initialize variables
n_ch = EEG['nbchan'][0, 0]
times = EEG['times'][0]
srate = EEG['srate'][0, 0]
n_pnts = EEG['pnts'][0, 0]
ch_names = [el[0] for el in EEG['chanlocs']['labels'][0]]
spat_maps = np.zeros((n_ch, len(freqs), len(filters), 2))
corr_data = np.full((len(freqs), len(filters), 2), np.nan)
snrs = np.full((len(freqs), len(filters), 2), np.nan)
plvs = np.full((len(freqs), len(filters), 2), np.nan)
xmin, xmax = EEG['xmin'].item(), EEG['xmax'].item()
t_idx = np.searchsorted(times, [xmin + 0.5, xmax - 0.5])

# Create an info object for topo plotting and SSD
info = create_info(ch_names=ch_names, sfreq=srate, ch_types='eeg')
info['subject_info'] = {'his_id': 'simulation'}
info.set_montage('standard_1020')

# Loop over frequencies
# Simulate EEG data
EEG = orig_EEG.copy()

# Create data time series
amp1 = 10 + 10 * filterFGx(np.random.randn(n_pnts), srate, 3, 10)[0]
freq_mod1 = detrend(10 * filterFGx(np.random.randn(n_pnts), srate, 3, 10)[0])
k1 = (dip_freq1 / srate) * 2 * np.pi / dip_freq1
signal1 = amp1 * np.sin(2 * np.pi * dip_freq1 * times + k1 * np.cumsum(freq_mod1))

amp2 = 10 + 10 * filterFGx(np.random.randn(n_pnts), srate, 3, 10)[0]
freq_mod2 = detrend(10 * filterFGx(np.random.randn(n_pnts), srate, 3, 10)[0])
k2 = (dip_freq2 / srate) * 2 * np.pi / dip_freq2
signal2 = amp2 * np.sin(2 * np.pi * dip_freq2 * times + k2 * np.cumsum(freq_mod2))

# Create dipole data
pwr_spec = filterFGx(
    (
        np.random.rand(n_pnts, lf.shape[2])
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
tmp_data = lf[:, orientation, :] @ data.T
EEG['data'] = tmp_data[:, t_idx[0] : t_idx[1] + 1]
EEG['pnts'] = EEG['data'].shape[1]
EEG['times'] = times[t_idx[0] : t_idx[1] + 1]

signal1 = signal1[0, t_idx[0] : t_idx[1] + 1]
signal2 = signal2[0, t_idx[0] : t_idx[1] + 1]

for fi, freq in enumerate([dip_freq1, dip_freq2]):
    if fi == 0:
        signal = signal1
    else:
        signal = signal2

    # Extract data for covariance matrix
    filt_data = filterFGx(EEG['data'], srate, freq, fwhm_filt)[0]
    filt_cov = (filt_data @ filt_data.T) / filt_data.shape[1]
    bb_cov = (EEG['data'] @ EEG['data'].T) / EEG['pnts']

    # Find frequency indices
    hz = np.linspace(0, srate, (np.diff(t_idx) + 1)[0])
    freq_idx = np.searchsorted(hz, freq)
    f_low = np.arange(np.searchsorted(hz, freq - 5), np.searchsorted(hz, freq - 1) + 1)
    f_high = np.arange(np.searchsorted(hz, freq + 1), np.searchsorted(hz, freq + 5) + 1)

    # GEDb
    # ==============================================================================
    filt_num = filters['GEDb']

    evals, ged_vecs = eigh(filt_cov, bb_cov)

    max_idx = np.argmax(evals)
    maps = filt_cov @ ged_vecs @ np.linalg.inv(ged_vecs.T @ filt_cov @ ged_vecs)
    maps = maps[:, max_idx]

    ged_data = (
        filterFGx(EEG['data'], srate, freq, fwhm_anal)[0].T @ ged_vecs[:, max_idx]
    )

    idx = np.argmax(np.abs(maps))
    spat_maps[:, fi, filt_num, 0] = maps * np.sign(maps[idx])

    corr_data[fi, filt_num, 0] = np.corrcoef(ged_data, signal)[0, 1] ** 2

    plvs[fi, filt_num, 0] = compute_plv(ged_data, signal)

    ged_data = EEG['data'].T @ ged_vecs[:, max_idx]
    f = np.abs(fft(ged_data) / EEG['pnts']) ** 2
    snrs[fi, filt_num, 0] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])


# NID
# ==============================================================================
filt_num = filters['NID']
print('NID')

df = 2
filter_order = 2
f_m = 1
f_n = 2
X = EEG['data']

X_ssd1, A1 = ssd_orig(EEG['data'], dip_freq1 - df, dip_freq1 + df, df, reduce=True)
X_ssd2, A2 = ssd_orig(EEG['data'], dip_freq2 - df, dip_freq2 + df, df, reduce=True)

X_stacked = np.vstack([X_ssd1, X_ssd2])

whiten = ['arbitrary-variance', 'unit-variance', False]
fun = ['logcosh', 'exp', 'cube']
# Evaluate all possible combinations of whiten and fun
combinations = [(w, f) for w in whiten for f in fun]
comb_idx = 0

# ica = FastICA(whiten=combinations[comb_idx][0], fun=combinations[comb_idx][1])
# ica.fit(X_stacked.T)
# A_ica = ica.mixing_

# Run ICA
channel_names = [f'EEG {i}' for i in range(X_stacked.shape[0])]  # Channel names
channel_types = ['eeg'] * X_stacked.shape[0]  # Assume all are EEG channels
info_ica = create_info(ch_names=channel_names, sfreq=srate, ch_types=channel_types)
raw = RawArray(X_stacked, info_ica)
ica = ICA(max_iter='auto')
ica.fit(raw)

A_ica = ica.mixing_matrix_

A_final1 = A1 @ A_ica[: A1.shape[1], :]
A_final2 = A2 @ A_ica[A1.shape[1] :, :]

idx = np.argmax(np.abs(A_final1[:, 0]))
spat_maps[:, 0, 5, 0] = A_final1[:, 0] * np.sign(A_final1[idx, 0])
corr_data[0, 5, 0] = np.corrcoef(X_ssd1[0, :], signal1)[0, 1] ** 2
plvs[0, 5, 0] = compute_plv(X_ssd1[0, :], signal1)
f = np.abs(fft(X_ssd1[0, :]) / EEG['pnts']) ** 2
# snrs[0, 5, 0] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

idx = np.argmax(np.abs(A_final2[:, 0]))
spat_maps[:, 1, 5, 0] = A_final2[:, 0] * np.sign(A_final2[idx, 0])
corr_data[1, 5, 0] = np.corrcoef(X_ssd2[0, :], signal2)[0, 1] ** 2
plvs[1, 5, 0] = compute_plv(X_ssd2[0, :], signal2)
f = np.abs(fft(X_ssd2[0, :]) / EEG['pnts']) ** 2
# snrs[1, 5, 0] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

# Find the indices of the frequencies
frequencies = freqs
freqs_to_plot = [np.abs(freqs - f).argmin() for f in frequencies]
filts_to_plot = [3, 5]

cmap = get_cmap('parula')
fig, axes = plt.subplots(
    len(frequencies),
    len(filts_to_plot),
    figsize=(8, 6),
)
axes = axes.ravel()

for fi in range(len(frequencies)):
    for i, filti in enumerate(filts_to_plot):
        ax = axes[fi * len(filts_to_plot) + i]

        # Extract the real part of the spatial map
        data = np.real(spat_maps[:, freqs_to_plot[fi], filti, 0])

        # Create the topomap
        plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            cmap=cmap,
        )

        # Set the title
        if fi == 0:
            ax.set_title(f'{list(filters.keys())[filti]}')
        if i == 0:
            ax.set_ylabel(f'{round(freqs[freqs_to_plot[fi]])} Hz')

fig.suptitle('No Noise')
plt.savefig(os.path.join('img', 'topo_no_noise.png'))
plt.show()

# %%
