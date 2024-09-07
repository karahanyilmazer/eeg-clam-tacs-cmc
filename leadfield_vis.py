# %%
# !%matplotlib inline
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.pyplot import close as close_plt
from mne.channels import make_standard_montage
from mne.datasets import fetch_fsaverage, sample
from mne.viz import plot_topomap
from numpy.fft import fft, ifft
from scipy.io import loadmat
from scipy.signal import butter, detrend, filtfilt, hilbert
from yaml import safe_dump, safe_load

from utils import filterFGx, get_base_dir, get_cmap, read_raw, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _ = get_base_dir(), get_cmap('parula'), set_style()
sys.path.insert(0, os.path.join(base_dir, 'eeg-classes'))

mne.set_log_level('INFO')

# %%
close_plt('all')

# Read in configuration values
with open('config.yaml', 'r') as file:
    config = safe_load(file)

subj = config['excluded_participants'][0]
print(subj)

t_tmin, t_max = config['cropping'][subj]
l_freq = config['l_freq']
h_freq = config['h_freq']
df = config['df'][subj]
orig_comp = config['orig_comp'][subj]
base_dir = config['base_dir_win'] if sys.platform == 'win32' else config['base_dir_mac']
img_folder = 'img'

raw = read_raw(subj)

# Get the number of time points and the sampling rate
srate = raw.info['sfreq']
data, t = raw.get_data(return_times=True)
n_pnts = len(t)

# %% Compute the leadfield matrix
# Create a standard info object with channel names from the montage
montage = make_standard_montage('standard_1020')  # Use the 10-20 system
info_sim = mne.create_info(ch_names=montage.ch_names, sfreq=srate, ch_types='eeg')
info_sim.set_montage(montage)

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

# Only pick the channels that are present in the raw data
fwd = mne.pick_channels_forward(fwd, include=raw.ch_names)
# Extract the leadfield matrix
lf = fwd['sol']['data']

print('Leadfield matrix shape:', lf.shape)  # (n_channels, n_dipoles)

# Source orientations (normals to the cortical surface if forward is fixed)
source_normals = fwd['source_nn']  # Normal vectors (n_sources x 3)

# Initialize an array to hold the normal-oriented lead field
lf_N = np.zeros((lf.shape[0], source_normals.shape[0]))

# Compute the lead field for dipoles oriented normal to the cortical surface
for i in range(3):
    # Multiply each leadfield component by the corresponding normal vectors
    lf_N += lf * source_normals[:, i]

# %%
plot = False
if plot:
    mne.viz.plot_alignment(
        info_sim,  # info_sim
        trans=trans,
        subject=subject,
        subjects_dir=fs_dir,
        src=src,
        eeg=['original', 'projected'],
    )

# %%
# Step 1: Find the index of the C3 channel in the forward solution
c3_idx = fwd['info']['ch_names'].index('C3')

# Step 2: Extract the leadfield matrix row corresponding to C3
c3_lf = lf[c3_idx, :]

# Step 3: Compute the magnitudes of the dipole projections
# Check if the forward solution has free orientation (3 components per dipole)
# Number of dipoles per hemisphere (assuming 1 source space)
n_dipoles = fwd['src'][0]['nuse']

if lf.shape[1] == 3 * n_dipoles * 2:
    # Free orientation case
    c3_magnitudes = np.sqrt(c3_lf[0::3] ** 2 + c3_lf[1::3] ** 2 + c3_lf[2::3] ** 2)
else:
    # Fixed orientation case
    c3_magnitudes = np.abs(c3_lf)


# Get the first three indices of the maximum projection
max_dip_idx = np.argsort(c3_magnitudes)[::-1]
max_proj_val = c3_magnitudes[max_dip_idx]

print(f'Index of C3 channel in the forward solution: {c3_idx}')
print(f'Dipole index with the maximum projection to C3: {max_dip_idx}')
print(f'Maximum projection value to C3: {max_proj_val}')
# %%
# for dip_loc in max_dip_idx[40:100]:
for dip_loc in [0]:
    # Simulate the data
    dipole_data = 1 * np.random.randn(lf_N.shape[1], 1000)
    # Add signal to the second half of the dataset
    dipole_data[dip_loc, 500:] = 15 * np.sin(2 * np.pi * 10 * np.arange(500) / srate)
    # Project dipole data to scalp electrodes
    data = lf_N @ dipole_data
    # Generate meaningless time series
    times = np.squeeze(np.arange(data.shape[1]) / srate)

    # Extract the location of 'C3'
    c3_loc = raw.info['chs'][c3_idx]['loc'][:2]  # x, y position

    # Plot the topomap
    fig, axs = plt.subplots(2, 1)

    plot_topomap(lf_N[:, dip_loc], raw.info, axes=axs[0], show=False, cmap=cmap)
    # Add the label 'C3' at the corresponding electrode position
    # axs[0].text(c3_loc[0], c3_loc[1], 'C3', color='black', ha='center', va='center')
    axs[0].plot(-4.75e-02, 0, 'o', color='red', markersize=10)

    axs[0].set_title(f'Signal dipole projection {dip_loc}')

    axs[1].plot(
        times,
        dipole_data[dip_loc, :] / np.linalg.norm(dipole_data[dip_loc, :]),
        linewidth=4,
        label='Dipole',
    )
    axs[1].plot(
        times,
        data[c3_idx, :] / np.linalg.norm(data[c3_idx, :]),
        linewidth=2,
        label='Electrode',
    )
    axs[1].legend()
    plt.show()
# %%
# Get the coordinates for the left and right hemisphere source space
lh_vertices = src[0]['rr'][src[0]['inuse'].astype(bool)]
rh_vertices = src[1]['rr'][src[1]['inuse'].astype(bool)]

# Combine left and right hemisphere locations into a single array
grid_locs = np.vstack((lh_vertices, rh_vertices))

# 3D Plot of dipole locations
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for all dipole locations
ax.scatter(grid_locs[:, 0], grid_locs[:, 1], grid_locs[:, 2], color='b', s=20)

# Highlight specific dipole location
ax.scatter(
    grid_locs[dip_loc, 0],
    grid_locs[dip_loc, 1],
    grid_locs[dip_loc, 2],
    color='r',
    marker='o',
    s=100,
)

plt.title('Brain Dipole Locations')
plt.show()
