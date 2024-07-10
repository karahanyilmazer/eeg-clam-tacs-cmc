# %%
# Import the FOOOF object
import matplotlib.pyplot as plt
import numpy as np
from fooof import FOOOF
from fooof.analysis import get_band_peak_fm
from fooof.bands import Bands
from scipy.signal import find_peaks

from utils import get_base_dir, get_cmap, get_P_TARGET, read_raw, set_fig_dpi, set_style

# Set figure and path settings
base_dir, cmap, _, _ = get_base_dir(), get_cmap('parula'), set_style(), set_fig_dpi()


# Function to find the range for a peak
def find_range(data, peak, threshold):
    left = peak
    while left > 0 and data[left] > threshold:
        left -= 1
    right = peak
    while right < len(data) - 1 and data[right] > threshold:
        right += 1

    return [left, right]


# %%
# Define the SSD parameters
l_freq, h_freq = 10, 14
df = 2
n_comps = 1

# Read in the data
raw = read_raw('FS_07')

# Apply the SSD
M, D, freqs, psd, pf = get_P_TARGET(
    raw, l_freq, h_freq, df, n_comps=n_comps, save=False
)

# %%
fm = FOOOF(max_n_peaks=2)
freq_range = [3, 30]
fm.fit(freqs, psd, freq_range)

fm.print_results()
fm.plot()
# %%
data = fm._peak_fit
# Find peaks
peaks, _ = find_peaks(data)

# Determine threshold to consider values as part of Gaussian (e.g., 1% of max value)
threshold = 0.01 * np.max(data)

# Find ranges for each peak
ranges = [find_range(data, peak, threshold) for peak in peaks]
ranges = fm.freqs[ranges]
ranges[:, 0] = np.floor(ranges[:, 0])
ranges[:, 1] = np.ceil(ranges[:, 1])

alpha_range = ranges[0]
beta_range = ranges[1]

# %%
# Define the SSD parameters
l_freq, h_freq = alpha_range
df = 2
n_comps = 1

# Read in the data
raw = read_raw('FS_07')

# Apply the SSD
M, D, freqs, psd, pf = get_P_TARGET(
    raw, l_freq, h_freq, df, n_comps=n_comps, save=False
)

# %%
# Define the SSD parameters
l_freq, h_freq = beta_range
df = 2
n_comps = 1

# Read in the data
raw = read_raw('FS_07')

# Apply the SSD
M, D, freqs, psd, pf = get_P_TARGET(
    raw, l_freq, h_freq, df, n_comps=n_comps, save=False
)
