# %% [markdown]
# ## PYTHON code accompanying the paper:
# ### A tutorial on generalized eigendecomposition for denoising, contrast enhancement, and dimension reduction in multichannel electrophysiology
#
# Mike X Cohen (mikexcohen@gmail.com)
#
# The files emptyEEG.mat, filterFGx.m, and topoplotindie.m need to be in
# the current directory.
# %%
# !%matplotlib inline
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
from filterFGxfun import filterFGx
from matplotlib.gridspec import GridSpec
from pytopo import topoplotIndie

from utils import filterFGx, get_base_dir, get_cmap, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()

# %%
# Mat file containing EEG, leadfield and channel locations
mat = sio.loadmat('emptyEEG')
lf = mat['lf'][0, 0]
EEG = mat['EEG'][0, 0]

dip_loc = 108

# Normal dipoles (normal to the surface of the cortex)
lf_gain = np.zeros((64, 2004))
for i in range(3):
    lf_gain += lf['Gain'][:, i, :] * lf['GridOrient'][:, i]

# Simulate the data
dip_data = 1 * np.random.randn(lf['Gain'].shape[2], 1000)
# Add signal to second half of dataset
dip_data[dip_loc, 500:] = 15 * np.sin(2 * np.pi * 10 * np.arange(500) / EEG['srate'])
# Project dipole data to scalp electrodes
EEG['data'] = lf_gain @ dip_data
# Meaningless time series
EEG['times'] = np.squeeze(np.arange(EEG['data'].shape[1]) / EEG['srate'])

# %%
# Plot brain dipoles
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the scatter points for all grid locations
ax.scatter(lf['GridLoc'][:, 0], lf['GridLoc'][:, 1], lf['GridLoc'][:, 2], marker='o')

# Highlight the dipole locations
ax.scatter(
    lf['GridLoc'][dip_loc, 0],
    lf['GridLoc'][dip_loc, 1],
    lf['GridLoc'][dip_loc, 2],
    c='tomato',
    marker='o',
    s=100,
)

# Set the title
ax.set_title('Brain Dipole Locations')

# Show the plot
plt.show()

# %%
_, axs = plt.subplots(2, 1)
topoplotIndie(lf_gain[:, dip_loc], EEG['chanlocs'], 'Signal Dipole Proj.', axs[0])

axs[1].plot(
    EEG['times'],
    dip_data[dip_loc, :] / np.linalg.norm(dip_data[dip_loc, :]),
    linewidth=4,
    label='Dipole',
)
axs[1].plot(
    EEG['times'],
    EEG['data'][30, :] / np.linalg.norm(EEG['data'][30, :]),
    linewidth=1,
    label='Electrode',
)
axs[1].legend()
axs[1].set_xlabel('Time (a.u.)')
axs[1].set_ylabel('Amplitude (a.u.)')

plt.tight_layout()
plt.show()

# %% Create covariance matrices

# Compute covariance matrix R is first half of data
tmpd = EEG['data'][:, :500]
covR = np.cov(tmpd)

# Compute covariance matrix S is second half of data
tmpd = EEG['data'][:, 500:]
covS = np.cov(tmpd)

# Plot the two covariance matrices
_, axs = plt.subplots(1, 3)

# S matrix
axs[0].imshow(covS, vmin=-1e6, vmax=1e6, cmap=cmap)
axs[0].set_title('S matrix')

# R matrix
axs[1].imshow(covR, vmin=-1e6, vmax=1e6, cmap=cmap)
axs[1].set_title('R matrix')

# R^{-1}S
axs[2].imshow(np.linalg.inv(covR) @ covS, vmin=-10, vmax=10, cmap=cmap)
axs[2].set_title('$R^{-1}S$ matrix')

plt.tight_layout()
plt.show()

# %% Dimension compression via PCA
# PCA
evals, evecs = scipy.linalg.eigh(covS + covR)

# Sort eigenvalues/vectors
s_idx = np.argsort(evals)[::-1]
evals = evals[s_idx]
evecs = evecs[:, s_idx]


# Plot the eigenspectrum
fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[2, 1])

# First subplot - combined A B and A C
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(evals / np.max(evals), 's-', markersize=10, markerfacecolor='k')
ax1.set_xlim([-0.5, 20.5])
ax1.set_title('PCA Eigenvalues')
ax1.set_xlabel('Component Number')
ax1.set_ylabel(r'Power Ratio (Norm. $\lambda$)')

# Filter forward model
filt_topo = evecs[:, 0]

# Eigenvector sign
se = np.argmax(np.abs(filt_topo))
filt_topo = filt_topo * np.sign(filt_topo[se])

# Plot the maps
topoplotIndie(lf_gain[:, dip_loc], EEG['chanlocs'], 'Truth Topomap', ax2)
topoplotIndie(filt_topo, EEG['chanlocs'], 'PCA Forward Model', ax3)

plt.show()

# Component time series is eigenvector as spatial filter for data
comp_ts = evecs[:, 0].T @ EEG['data']

# Normalize time series (for visualization)
dipl_ts = dip_data[dip_loc, :] / np.linalg.norm(dip_data[dip_loc, :])
comp_ts = comp_ts / np.linalg.norm(comp_ts)
chan_ts = EEG['data'][30, :] / np.linalg.norm(EEG['data'][30, :])

# Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(EEG['times'], 0.3 + dipl_ts, label='Truth')
plt.plot(EEG['times'], 0.15 + chan_ts, label='EEG Channel')
plt.plot(EEG['times'], comp_ts, label='PCA Time Series')
plt.xlabel('Time (a.u.)')
plt.legend(loc='lower left')
plt.yticks([])
plt.show()

# %% Source separation via GED
# GED
evals, evecs = scipy.linalg.eigh(covS, covR)

# Sort eigenvalues/vectors
s_idx = np.argsort(evals)[::-1]
evals = evals[s_idx]
evecs = evecs[:, s_idx]


# Plot the eigenspectrum
fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[2, 1])

# First subplot - combined A B and A C
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(evals / np.max(evals), 's-', markersize=10, markerfacecolor='k')
ax1.set_xlim([-0.5, 20.5])
ax1.set_title('GED Eigenvalues')
ax1.set_xlabel('Component Number')
ax1.set_ylabel(r'Power Ratio (Norm. $\lambda$)')

# Filter forward model
filt_topo = evecs[:, 0].T @ covS  # Small change comapred to PCA

# Eigenvector sign
se = np.argmax(np.abs(filt_topo))
filt_topo = filt_topo * np.sign(filt_topo[se])

# Plot the maps
topoplotIndie(lf_gain[:, dip_loc], EEG['chanlocs'], 'Truth Topomap', ax2)
topoplotIndie(filt_topo, EEG['chanlocs'], 'GED Forward Model', ax3)

plt.show()

# Component time series is eigenvector as spatial filter for data
comp_ts = evecs[:, 0].T @ EEG['data']

# Normalize time series (for visualization)
dipl_ts = dip_data[dip_loc, :] / np.linalg.norm(dip_data[dip_loc, :])
comp_ts = comp_ts / np.linalg.norm(comp_ts)
chan_ts = EEG['data'][30, :] / np.linalg.norm(EEG['data'][30, :])

# Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(EEG['times'], 0.3 + dipl_ts, label='Truth')
plt.plot(EEG['times'], 0.15 + chan_ts, label='EEG Channel')
plt.plot(EEG['times'], comp_ts, label='GED Time Series')
plt.xlabel('Time (a.u.)')
plt.legend(loc='lower left')
plt.yticks([])
plt.show()

# %% [markdown]
# ## Example GED in richer data

# The above simulation is overly simplistic. The goal of
# this section is to simulate data that shares more
# characteristics to real EEG data, including non-sinusoidal
# rhythms, background noise, and multiple trials.
#
# This code will simulate resting-state that has been segmented
# into 2-second non-overlapping epochs.

# %% Simulate the data

# Signal parameters in Hz
peak_freq = 10  # Alpha peak frequency
fwhm = 5  # Full-width at half-maximum around the alpha peak


# EEG parameters for the simulation
EEG['srate'] = 500  # Sampling rate in Hz
EEG['pnts'] = 2 * EEG['srate']  # Each data segment is 2 seconds
EEG['trials'] = 50
EEG['data'] = np.zeros((EEG['nbchan'][0][0], EEG['pnts'], EEG['trials']))

# Create frequency-domain Gaussian
hz = np.linspace(0, EEG['srate'], EEG['pnts'])
s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # Normalized width
x = hz - peak_freq  # Shifted frequencies
fg = np.exp(-0.5 * (x / s) ** 2)  # Gaussian


# Loop over trials and generate data
for trial_i in range(EEG['trials']):

    # Random Fourier coefficients
    fc = np.random.rand(EEG['pnts']) * np.exp(
        1j * 2 * np.pi * np.random.rand(1, EEG['pnts'])
    )

    # Taper with the Gaussian
    fc = fc * fg

    # Back to time domain to get the source activity
    source_ts = 2 * np.real(np.fft.ifft(fc)) * EEG['pnts']

    # Simulate dipole data: all noise and replace target dipole with source_ts
    dip_data = np.random.randn(np.shape(lf_gain)[1], EEG['pnts'])
    dip_data[dip_loc, :] = 0.5 * source_ts
    # Note: the source time series has low amplitude to highlight the
    # sensitivity of GED. Increasing this gain to, e.g., 1 will show
    # accurate though noiser reconstruction in the channel data.

    # Now project the dipole data through the forward model to the electrodes
    EEG['data'][:, :, trial_i] = lf_gain @ dip_data


# %%
# Topoplot of alpha power
ch_pwr = np.abs(np.fft.fft(EEG['data'], axis=1)) ** 2
ch_pwr_avg = np.mean(ch_pwr, axis=2)

# Vector of frequencies
hz = np.linspace(0, EEG['srate'] / 2, EEG['pnts'] // 2 + 1)

# %%
# Create a covariance tensor (one covmat per trial)

# Filter the data around 10 Hz
alpha_filt = copy.deepcopy(EEG['data'])
for ti in range(int(EEG['trials'])):
    alpha_filt[:, :, ti] = filterFGx(EEG['data'][:, :, ti], EEG['srate'], 10, 4)[0]

# Initialize covariance matrices (one for each trial)
all_covS = np.zeros((EEG['trials'], EEG['nbchan'][0][0], EEG['nbchan'][0][0]))
all_covR = np.zeros((EEG['trials'], EEG['nbchan'][0][0], EEG['nbchan'][0][0]))


# Loop over trials (data segments) and compute each covariance matrix
for trial_i in range(EEG['trials']):

    # Cut out a segment
    tmp_data = alpha_filt[:, :, trial_i]

    # Mean-center
    tmp_data = tmp_data - np.mean(tmp_data, axis=1, keepdims=True)

    # Add to S tensor
    all_covS[trial_i, :, :] = tmp_data @ tmp_data.T / EEG['pnts']

    # Repeat for broadband data
    tmp_data = EEG['data'][:, :, trial_i]
    tmp_data = tmp_data - np.mean(tmp_data, axis=1, keepdims=True)
    all_covR[trial_i, :, :] = tmp_data @ tmp_data.T / EEG['pnts']

# %%
# Illustration of cleaning covariance matrices

# Clean R
meanR = np.mean(all_covR, axis=0)  # Average covariance
dists = np.zeros(EEG['trials'])  # Vector of distances to mean
for segi in range(EEG['trials']):
    r = all_covR[segi, :, :]
    # Euclidean distance
    dists[segi] = np.sqrt(np.sum((r.reshape(1, -1) - meanR.reshape(1, -1)) ** 2))

# Compute z-scored distances
distsZ = (dists - np.mean(dists)) / np.std(dists)

# Finally, average trial-covariances together, excluding outliers
covR = np.mean(all_covR[distsZ < 3, :, :], axis=0)

# Normally you'd repeat the above for S; ommitted here for simplicity
covS = np.mean(all_covS, axis=0)

# %%
# Now for the GED

# NOTE: You can test PCA on these data by using only covS, or only covR,
#       in the eig() function.

# Eig and sort
evals, evecs = scipy.linalg.eigh(covS, covR)
s_idx = np.argsort(evals)[::-1]
evals = evals[s_idx]
evecs = evecs[:, s_idx]


# Compute the component time series for the multiplication
# The data need to be reshaped into 2D
data_2d = np.reshape(EEG['data'], (EEG['nbchan'][0][0], -1), order='F')
comps = evecs[:, 0].T @ data_2d
# and then reshaped back into trials
comps = np.reshape(comps, (EEG['pnts'], EEG['trials']), order='F')

# Power spectrum
comp_pwr = np.abs(np.fft.fft(comps, axis=0)) ** 2
comp_pwr_avg = np.mean(comp_pwr, axis=1)

# Component map
comp_map = evecs[:, 0].T @ covS
# Flip map sign
se = np.argmax(np.abs(comp_map))
comp_map = comp_map * np.sign(comp_map[se])


# %% Visualization
fig = plt.figure()
gs = GridSpec(3, 2, width_ratios=[1, 1.5])

# First subplot - combined A B and A C
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])

topoplotIndie(lf_gain[:, dip_loc], EEG['chanlocs'], 'Truth Topomap', ax2)

ax1.plot(evals, 'ks-', markersize=10, markerfacecolor='r')
ax1.set_xlim([0, 20.5])
ax1.set_title('GED Scree Plot')
ax1.set_xlabel('Component Number')
ax1.set_ylabel(r'Power Ratio ($\lambda$)')
# Note that the max eigenvalue is <1,
# because R has more overall energy than S.

# GED component
topoplotIndie(comp_map, EEG['chanlocs'], 'Alpha Component', ax3)

# channel 10 Hz power
where_10 = np.argmin(np.abs(hz - 10))
topoplotIndie(ch_pwr_avg[:, where_10], EEG['chanlocs'], 'Electr. Power (10 Hz)', ax4)
plt.tight_layout()
plt.show()


# Spectra
plt.figure(figsize=(10, 4))
plt.plot(
    hz,
    comp_pwr_avg[: len(hz)] / np.max(comp_pwr_avg[: len(hz)]),
    'r',
    linewidth=3,
    label='Component',
)
plt.plot(
    hz,
    ch_pwr_avg[30, : len(hz)] / np.max(ch_pwr_avg[30, : len(hz)]),
    'k',
    linewidth=3,
    label='Electrode 31',
)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (Norm to Max. Power)')
plt.xlim([0, 80])
plt.legend(loc='lower right')
plt.show()

# %%
