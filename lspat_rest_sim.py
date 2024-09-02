#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add motor oscillations on top of the resting state data.

@author: Karahan Yilmazer
"""

# %%
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

import numpy as np
from matplotlib.pyplot import close as close_plt
from mne import set_log_level
from yaml import safe_dump, safe_load

from filterFGxfun import filterFGx
from utils import get_base_dir, get_cmap, read_raw, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()
sys.path.insert(0, os.path.join(base_dir, 'eeg-classes'))
from src.source_space.SSD import SSD

set_log_level('WARNING')

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

# Read in the data
raw = read_raw(subj)
raw.compute_psd().plot()
raw.plot(scalings='auto')

ssd = SSD()
ssd.fit(raw, l_freq, h_freq, df)
ssd.plot(n_comps=5, img_folder=img_folder, save=False)
ssd.fit_fooof(config, plot=True)
alpha, beta, peaks = ssd.adjust_freq_bands(config['gauss_thr'][subj], plot=True)

# %%
raw = read_raw(subj)
c3_idx = raw.ch_names.index('C3')
n_samples = raw.times.shape[0]  # Total number of samples
alpha_freq = 12

# Generate time vector
time = np.arange(n_samples) / sfreq

# Generate synthetic alpha oscillation
alpha_oscillation = 5e-7 * np.sin(2 * np.pi * alpha_freq * time)

raw._data[c3_idx] += alpha_oscillation
raw.compute_psd().plot()
raw.plot(scalings='auto')

ssd = SSD()
ssd.fit(raw, l_freq, h_freq, df)
ssd.plot(n_comps=5, img_folder=img_folder, save=False)
# ssd.fit_fooof(config, plot=True)
# alpha, beta, peaks = ssd.adjust_freq_bands(config['gauss_thr'][subj], plot=True)
# %%
alpha_freq = 10  # Center frequency of the alpha wave (10 Hz is typical for alpha)
alpha_fwhm = 2  # Bandwidth for the alpha oscillation
amp_mod_freq = 1  # Frequency for amplitude modulation (optional)
amp_mod_fwhm = 0.5  # Bandwidth for amplitude modulation
srate = raw.info['sfreq']

# Create amplitude modulation signal
amp_mod = (
    1
    + 0.5 * filterFGx(np.random.randn(n_samples), srate, amp_mod_freq, amp_mod_fwhm)[0]
)

# Create frequency modulation signal
freq_mod = detrend(
    0.1 * filterFGx(np.random.randn(n_samples), srate, alpha_freq, alpha_fwhm)[0]
)

# Generate alpha oscillation with amplitude and frequency modulation
k_alpha = (alpha_freq / srate) * 2 * np.pi / alpha_freq
alpha_wave = amp_mod * np.sin(
    2 * np.pi * alpha_freq * times + k_alpha * np.cumsum(freq_mod)
)

# Optional: Scale the alpha wave amplitude to match typical EEG data
alpha_wave *= 0.0001  # Adjust scaling as needed

# Find the index of the C3 electrode
c3_idx = raw.ch_names.index('C3')

# Extract the original resting-state data for C3
original_data, times = raw.get_data(picks='C3', return_times=True)

# Inject the simulated alpha wave into the C3 electrode data
modified_data = original_data[0] + alpha_wave

# Replace the C3 data in the Raw object with the modified data
raw._data[c3_idx, :] = modified_data

# %%
