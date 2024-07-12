# %%
# !%matplotlib inline
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
subj = 'FS_03'
raw = read_raw(subj)

with open('config.yaml', 'r') as file:
    config = safe_load(file)

# Initialize the SSD
l_freq = config['l_freq']
h_freq = config['h_freq']
df = config['df']

ssd = SSD()
ssd.fit(raw, l_freq, h_freq, df)
ssd.plot(n_comps=1, prefix='Alpha', save=False)
ssd.fit_fooof(config)
alpha, beta = ssd.adjust_freq_bands(config['gauss_thr'][subj])

# %%
config['alpha_range'] = alpha
config['beta_range'] = beta

ssd_alpha = SSD()
ssd_alpha.fit(raw, alpha[0], alpha[1], df)
ssd_alpha.plot(n_comps=1, prefix='Alpha', save=False)

ssd_beta = SSD()
ssd_beta.fit(raw, beta[0], beta[1], df)
ssd_beta.plot(n_comps=1, prefix='Beta', save=False)
# %%
