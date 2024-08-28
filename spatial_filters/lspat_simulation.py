# %%
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import filterFGx, get_base_dir, get_cmap

sys.path.insert(0, os.path.join(get_base_dir(), 'eeg-classes'))

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from mne import create_info
from mne.io import RawArray
from mne.viz import plot_topomap
from numpy.fft import fft, ifft
from scipy.io import loadmat
from scipy.linalg import eigh, pinv, toeplitz
from scipy.signal import detrend, hilbert
from sklearn.decomposition import FastICA
from src.source_space.SSD import SSD

plt.rcParams.update({'figure.dpi': 300})
plt.style.use(['science', 'no-latex'])

# %%
# Load data from the mat file
mat = loadmat('emptyEEG.mat')
EEG = mat['EEG'][0, 0]
lf = mat['lf'][0, 0][2]

orig_EEG = EEG.copy()

# Filter parameters
freqs = np.logspace(np.log10(4), np.log10(80), 10)
fwhm_filt = 2
fwhm_anal = 5

# Indices of dipole locations
dip_loc1 = 93
dip_loc2 = 204
orientation = 0  # 0 for "EEG" and 1 for "MEG"

# Define the filters to evaluate
filters = {'Best Electrode': 0, 'PCA': 1, 'JD': 2, 'GEDb': 3, 'SSD': 4}

# Initialize variables
n_ch = EEG['nbchan'][0, 0]
times = EEG['times'][0]
srate = EEG['srate'][0, 0]
n_pnts = EEG['pnts'][0, 0]
ch_names = [el[0] for el in EEG['chanlocs']['labels'][0]]
spat_maps = np.zeros((n_ch, len(freqs), len(filters), 2))
corr_data = np.full((len(freqs), len(filters), 2), np.nan)
snrs = np.full((len(freqs), len(filters), 2), np.nan)
xmin, xmax = EEG['xmin'].item(), EEG['xmax'].item()
t_idx = np.searchsorted(times, [xmin + 0.5, xmax - 0.5])

# Create an info object for topo plotting and SSD
info = create_info(ch_names=ch_names, sfreq=srate, ch_types='eeg')
info['subject_info'] = {'his_id': 'simulation'}
info.set_montage('standard_1020')

# %%
# Loop over frequencies
for fi, freq in enumerate(freqs):
    # Simulate EEG data
    EEG = orig_EEG.copy()

    dip_freq1 = freq
    dip_freq2 = dip_freq1 + np.random.rand() * 5 + 1

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
    tmp_data = (data @ lf[:, orientation, :].T).T
    EEG['data'] = tmp_data[:, t_idx[0] : t_idx[1] + 1]
    EEG['pnts'] = EEG['data'].shape[1]
    EEG['times'] = times[t_idx[0] : t_idx[1] + 1]

    signal1 = signal1[0, t_idx[0] : t_idx[1] + 1]

    # EEG['data'] = loadmat('data.mat')['data']
    # filt_data = loadmat('filtdat.mat')['filtdat']
    # filt_cov = loadmat('filtcov.mat')['filtcov']
    # bb_cov = loadmat('bbcov.mat')['bbcov']
    # signal1 = loadmat('signal1.mat')['signal1']

    for noise_i in range(2):
        if noise_i == 1:
            # Replace O1 with pure noise
            o1_idx = ch_names.index('O1')
            m = np.mean(EEG['data'][o1_idx, :])
            v = np.var(EEG['data'][o1_idx, :])
            EEG['data'][o1_idx, :] = m + v * np.random.randn(EEG['pnts'])

        # Extract data for covariance matrix
        filt_data = filterFGx(EEG['data'], srate, freq, fwhm_filt)[0]
        filt_cov = (filt_data @ filt_data.T) / filt_data.shape[1]
        bb_cov = (EEG['data'] @ EEG['data'].T) / EEG['pnts']

        # Find frequency indices
        hz = np.linspace(0, srate, (np.diff(t_idx) + 1)[0])
        freq_idx = np.searchsorted(hz, freq)
        f_low = np.arange(
            np.searchsorted(hz, freq - 5), np.searchsorted(hz, freq - 1) + 1
        )
        f_high = np.arange(
            np.searchsorted(hz, freq + 1), np.searchsorted(hz, freq + 5) + 1
        )

        # Best-electrode
        # ==============================================================================
        filt_num = filters['Best Electrode']

        el_pwr = np.abs(hilbert(filt_data, axis=0)) ** 2
        max_el = np.argmax(np.mean(el_pwr, axis=1))

        spat_maps[:, fi, filt_num, noise_i] = np.mean(el_pwr, axis=1)

        tc_data = filterFGx(EEG['data'][max_el, :], srate, freq, fwhm_anal)[0]
        corr_data[fi, filt_num, noise_i] = np.corrcoef(tc_data, signal1)[0, 1] ** 2

        f = np.abs(fft(EEG['data'][max_el, :]) / EEG['pnts']) ** 2
        snrs[fi, filt_num, noise_i] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

        # PCA
        # ==============================================================================
        filt_num = filters['PCA']

        evals, pca_vecs = eigh(filt_cov)
        pca_data = (filterFGx(EEG['data'], srate, freq, fwhm_anal)[0].T @ pca_vecs).T
        fft_pwr = np.abs(fft(pca_data, axis=1)) ** 2
        best_comp = np.argmax(fft_pwr[:, np.searchsorted(hz, freq)])
        maps = pca_vecs[:, best_comp]

        idx = np.argmax(np.abs(maps))
        spat_maps[:, fi, filt_num, noise_i] = maps * np.sign(maps[idx])

        corr_data[fi, filt_num, noise_i] = (
            np.corrcoef(pca_data[best_comp, :], signal1)[0, 1] ** 2
        )

        pca_data = EEG['data'].T @ pca_vecs[:, idx]
        f = np.abs(fft(pca_data) / EEG['pnts']) ** 2
        snrs[fi, filt_num, noise_i] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

        # JDfilt
        # ==============================================================================
        filt_num = filters['JD']

        evalsO, evecsO = eigh(bb_cov)
        evalsO = np.diag(evalsO)
        sphere_data = (EEG['data'].T @ evecsO @ np.sqrt(pinv(evalsO))).T

        bias_filt = toeplitz(np.sin(2 * np.pi * freq * EEG['times']))
        zbar = bias_filt @ sphere_data.T
        evalsF, evecsF = eigh(zbar.T @ zbar / len(zbar))

        jdw = evecsO @ np.sqrt(pinv(evalsO)) @ evecsF
        jd_maps = pinv(jdw).T

        max_comp = np.argmax(evalsF)
        idx = np.argmax(np.abs(jd_maps[:, max_comp]))
        spat_maps[:, fi, filt_num, noise_i] = jd_maps[:, max_comp] * np.sign(
            jd_maps[idx, max_comp]
        )

        # Apply spatial filter to data
        jd_data = filterFGx(EEG['data'], srate, freq, fwhm_anal)[0].T @ jdw[:, max_comp]

        corr_data[fi, filt_num, noise_i] = np.corrcoef(jd_data, signal1)[0, 1] ** 2

        jd_data = EEG['data'].T @ jdw[:, max_comp]
        f = np.abs(fft(jd_data) / EEG['pnts']) ** 2
        snrs[fi, filt_num, noise_i] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

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
        spat_maps[:, fi, filt_num, noise_i] = maps * np.sign(maps[idx])

        corr_data[fi, filt_num, noise_i] = np.corrcoef(ged_data, signal1)[0, 1] ** 2

        ged_data = EEG['data'].T @ ged_vecs[:, max_idx]
        f = np.abs(fft(ged_data) / EEG['pnts']) ** 2
        snrs[fi, filt_num, noise_i] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

        # SSD
        # ==============================================================================
        filt_num = filters['SSD']

        ssd = SSD()
        raw = RawArray(EEG['data'], info)
        ssd.fit(raw, freq - 2, freq + 2, 2)

        idx = np.argmax(np.abs(ssd.M[:, 0]))
        spat_maps[:, fi, filt_num, noise_i] = ssd.M[:, 0] * np.sign(ssd.M[idx, 0])
        corr_data[fi, filt_num, noise_i] = (
            np.corrcoef(ssd.X_ssd[:, 0], signal1)[0, 1] ** 2
        )
        f = np.abs(fft(ssd.X_ssd[:, 0]) / EEG['pnts']) ** 2
        snrs[fi, filt_num, noise_i] = f[freq_idx] / np.mean(f[np.r_[f_low, f_high]])

# %%
# Assuming freqs, spatmaps are already defined
all_corrs = np.zeros((len(freqs), 2))

for fi in range(len(freqs)):
    for noisei in range(2):
        # Extract the real part of spatmaps
        real_spat_maps = np.real(spat_maps[:, fi, :, noisei])

        # Compute the correlation matrix
        corr_matrix = np.corrcoef(real_spat_maps, rowvar=False)

        # Extract the upper triangular part of the correlation matrix, excluding the diagonal
        allcs = np.triu(corr_matrix, k=1).flatten()

        # Remove zeros and compute the square
        allcs = allcs[allcs != 0] ** 2

        # Filter values that are finite and less than 1
        valid_allcs = allcs[(allcs < 1) | np.isfinite(allcs)]

        # Compute the mean of the valid correlations
        all_corrs[fi, noisei] = np.mean(valid_allcs)

# %%
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

for i in range(2):
    axs[i].plot(
        freqs, corr_data[:, :, i], 's-', linewidth=2, markersize=10, markerfacecolor='w'
    )
    if i == 1:
        axs[i].set_title('O1 Noise')
        axs[i].set_xlabel('Frequency (Hz)')
    else:
        axs[i].set_title('No Noise')
    axs[i].set_ylabel('Fit to Signal ($R^2$)')
    axs[i].set_xlim([freqs[0] - 0.1, freqs[-1] + 1])
    axs[i].set_ylim([0, 1])
    axs[i].set_xscale('log')
    axs[i].legend(filters.keys())

plt.savefig(os.path.join('img', 'r2_fit.png'))
plt.show()

# %%
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

for i in range(2):
    # Plotting the SNR values
    axs[i].plot(
        freqs, snrs[:, :, i], 's-', linewidth=2, markersize=10, markerfacecolor='w'
    )

    if i == 1:
        axs[i].set_title('O1 Noise')
        axs[i].set_xlabel('Frequency (Hz)')
    else:
        axs[i].set_title('No Noise')

    axs[i].set_ylabel('SNR')
    axs[i].set_xlim([freqs[0] - 0.1, freqs[-1] + 1])
    axs[i].set_ylim([0, 1.3 * np.max(snrs)])
    axs[i].set_xscale('log')

    axs[i].legend(filters.keys())

plt.savefig(os.path.join('img', 'snr.png'))
plt.show()

# %%
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

mean_corr = np.nanmean(corr_data, axis=0)
mean_snrs = np.nanmean(snrs, axis=0)

# Number of groups and bars per group
n_groups, n_bars = mean_corr.shape

# Create an array for the x-axis
index = np.arange(n_groups)

# Set the bar width
bar_width = 0.35

# Plot each set of bars
for i in range(n_bars):
    axs[0].bar(index + i * bar_width, mean_corr[:, i], bar_width)
    axs[1].bar(index + i * bar_width, mean_snrs[:, i], bar_width)

# Set x-axis labels and legend
axs[0].set_xticks(index + bar_width / 2 * (n_bars - 1))
axs[0].set_xticklabels(filters.keys())
axs[0].legend(['No Noise', 'O1 Noise'])
axs[0].set_ylim(0, 1)
axs[0].set_ylabel('Fit to Simulated Signal ($R^2$)')

axs[1].set_xticks(index + bar_width / 2 * (n_bars - 1))
axs[1].set_xticklabels(filters.keys())
axs[1].legend(['No Noise', 'O1 Noise'])
axs[1].set_ylabel('SNR')

plt.savefig(os.path.join('img', 'bar_plot.png'))
plt.show()

# %%
plt.figure(figsize=(8, 4))

plt.plot(freqs, all_corrs, '-s', linewidth=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Inter-Map Correlations')
plt.xlim([freqs[0] - 0.1, freqs[-1] + 1])
plt.xscale('log')
plt.legend(['No Noise', 'O1 Noise'])

plt.savefig(os.path.join('img', 'intermap_corr.png'))
plt.show()

# %%
# Find the indices of the frequencies
frequencies = np.array([3, 9, 20, 40, 70])
freqs_to_plot = [np.abs(freqs - f).argmin() for f in frequencies]

cmap = get_cmap('parula')
fig, axes = plt.subplots(
    len(frequencies),
    len(filters),
    figsize=(len(frequencies) * 2, len(filters) * 2),
)

for fi in range(len(frequencies)):
    for filti in range(len(filters)):
        ax = axes[fi, filti]

        # Extract the real part of the spatial map
        data = np.real(spat_maps[:, freqs_to_plot[fi], filti, 0])

        # Create the topomap
        plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap=cmap,
        )

        # Set the title
        if fi == 0:
            ax.set_title(f'{list(filters.keys())[filti]}')
        if filti == 0:
            ax.set_ylabel(f'{round(freqs[freqs_to_plot[fi]])} Hz')

fig.suptitle('No Noise')
plt.savefig(os.path.join('img', 'topo_no_noise.png'))
plt.show()

# %%
cmap = get_cmap('parula')
fig, axes = plt.subplots(
    len(frequencies),
    len(filters),
    figsize=(len(frequencies) * 2, len(filters) * 2),
)

for fi in range(len(frequencies)):
    for filti in range(len(filters)):
        ax = axes[fi, filti]

        data = np.real(spat_maps[:, freqs_to_plot[fi], filti, 1])
        plot_topomap(data, info, axes=ax, show=False, contours=0, cmap=cmap)

        if fi == 0:
            ax.set_title(f'{list(filters.keys())[filti]}')
        if filti == 0:
            ax.set_ylabel(f'{round(freqs[freqs_to_plot[fi]])} Hz')

fig.suptitle('O1 Noise')
plt.savefig(os.path.join('img', 'topo_o1_noise.png'))
plt.show()

# %%
