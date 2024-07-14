#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adjust the frequency borders for the alpha and beta bands using FOOOF fits.

@author: Karahan Yilmazer
"""

# %%
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

from yaml import safe_dump, safe_load

from utils import get_base_dir, get_cmap, read_raw, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()
sys.path.insert(0, os.path.join(base_dir, 'eeg-classes'))
from src.source_space.SSD import SSD

# %%
# Read in the data
subj = 'FS_07'
raw = read_raw(subj)

# Read in configuration values
with open('config.yaml', 'r') as file:
    config = safe_load(file)

img_folder = 'img'
df = config['df']
l_freq = config['l_freq']
h_freq = config['h_freq']
orig_comp = config['orig_comp'][subj]
alpha_comp = config['alpha_comp'][subj]
beta_comp = config['beta_comp'][subj]

# Apply the SSD
ssd = SSD()
ssd.fit(raw, l_freq, h_freq, df)
ssd.plot(comp_idx=orig_comp, img_folder=img_folder, save=False)
ssd.fit_fooof(config, plot=False)
alpha, beta, peaks = ssd.adjust_freq_bands(config['gauss_thr'][subj], plot=False)

# Store the resulting frequency parameters
config['alpha_range'][subj] = [float(freq) for freq in alpha]
config['beta_range'][subj] = [float(freq) for freq in beta]
config['alpha_peak'][subj] = float(ssd.fm.freqs[peaks[0]])
config['beta_peak'][subj] = float(ssd.fm.freqs[peaks[0]])

with open('config.yaml', 'w') as file:
    safe_dump(config, file, sort_keys=False)

# %%
# Fit the SSD again on individualized alpha and beta ranges
ssd_alpha = SSD()
ssd_alpha.fit(raw, alpha[0], alpha[1], df)
ssd_alpha.plot(comp_idx=alpha_comp, prefix='Alpha', img_folder=img_folder, save=False)

ssd_beta = SSD()
ssd_beta.fit(raw, beta[0], beta[1], df)
ssd_beta.plot(comp_idx=beta_comp, prefix='Beta', img_folder=img_folder, save=False)

# %%
