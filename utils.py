import sys
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.colors import LinearSegmentedColormap
from mne.channels import make_standard_montage
from mne.io import read_raw_brainvision
from scipy.fft import fft, ifft
from scipy.io import loadmat
from scipy.signal import hilbert
from yaml import safe_load

curr_dir = dirname(__file__)


def get_base_dir():
    # Load the config file
    f_name = join(curr_dir, 'preprocessing_parameters.yaml')
    with open(f_name, 'r') as file:
        config = safe_load(file)

    # Get the base directory
    platform = sys.platform
    if platform == 'win32':
        base_dir = config['base_dir_win']
    elif platform == 'darwin':
        base_dir = config['base_dir_mac']

    return base_dir


def set_style(notebook=True, grid=False):
    # Set the style to science
    args = ['science', 'no-latex']
    if grid:
        args.append('grid')
    if notebook:
        args.append('notebook')
    plt.style.use(args)


def set_fig_dpi():
    # Get the current OS
    platform = sys.platform
    if platform == 'win32':
        # Set the figure dpi to 260
        plt.matplotlib.rcParams['figure.dpi'] = 260


def get_cmap(name):
    # Load the (parula) cmap
    f_name = join(curr_dir, f'{name}.yaml')
    with open(f_name, 'r') as file:
        cmap = safe_load(file)['cmap']
        cmap = LinearSegmentedColormap.from_list(name, cmap)
    return cmap


def filterFGx(data, srate, f, fwhm, show_plot=False):
    """
    Narrow-band filter via frequency-domain Gaussian

    Parameters
    ----------
    data : array
        Input data, 1D or 2D (channels x time).
    srate : float
        Sampling rate in Hz.
    f : float
        Peak frequency of filter.
    fwhm : float
        Standard deviation of filter, defined as full-width at half-maximum in Hz.
    show_plot : bool, optional
        Set to True to show the frequency-domain filter shape.

    Returns
    -------
    filt_data : array
        Filtered data.
    emp_vals : list
        The empirical frequency and FWHM (in Hz and in ms).
    """

    # Input check
    if data.shape[0] > data.shape[1]:
        # raise ValueError(
        #     'Data dimensions may be incorrect. Data should be channels x time.'
        # )
        pass

    if (f - fwhm) < 0:
        # raise ValueError('Increase frequency or decrease FWHM.')
        pass

    if fwhm <= 0:
        raise ValueError('FWHM must be greater than 0.')

    # Frequencies
    hz = np.linspace(0, srate, data.shape[1])

    # Create Gaussian
    s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # Normalized width
    x = hz - f  # Shifted frequencies
    fx = np.exp(-0.5 * (x / s) ** 2)  # Gaussian
    fx = fx / np.abs(np.max(fx))  # Gain-normalized

    # Filter
    filt_data = 2 * np.real(ifft(fft(data, axis=1) * fx, axis=1))

    # Compute empirical frequency and standard deviation
    idx = np.argmin(np.abs(hz - f))
    emp_vals = [
        hz[idx],
        hz[idx - 1 + np.argmin(np.abs(fx[idx:] - 0.5))]
        - hz[np.argmin(np.abs(fx[:idx] - 0.5))],
    ]

    # Also temporal FWHM
    tmp = np.abs(hilbert(np.real(np.fft.fftshift(ifft(fx)))))
    tmp = tmp / np.max(tmp)
    tx = np.arange(data.shape[1]) / srate
    idxt = np.argmax(tmp)
    emp_vals.append(
        (
            tx[idxt - 1 + np.argmin(np.abs(tmp[idxt:] - 0.5))]
            - tx[np.argmin(np.abs(tmp[:idxt] - 0.5))]
        )
        * 1000
    )

    # Inspect the Gaussian (turned off by default)
    if show_plot:
        plt.figure()
        plt.subplot(211)
        plt.plot(hz, fx, 'o-')
        plt.plot(
            [
                hz[np.argmin(np.abs(fx[:idx] - 0.5))],
                hz[idx - 1 + np.argmin(np.abs(fx[idx:] - 0.5))],
            ],
            [
                fx[np.argmin(np.abs(fx[:idx] - 0.5))],
                fx[idx - 1 + np.argmin(np.abs(fx[idx:] - 0.5))],
            ],
            'k--',
        )
        plt.xlim([max(f - 10, 0), f + 10])
        plt.title(
            f'Requested: {f:.2f}, {fwhm:.2f} Hz; Empirical: {emp_vals[0]:.2f}, {emp_vals[1]:.2f} Hz'
        )
        plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude Gain')

        plt.subplot(212)
        tmp1 = np.real(np.fft.fftshift(ifft(fx)))
        tmp1 = tmp1 / np.max(tmp1)
        tmp2 = np.abs(hilbert(tmp1))
        plt.plot(tx, tmp1, tx, tmp2)
        plt.xlabel('Time (s)'), plt.ylabel('Amplitude Gain')

        plt.tight_layout()
        plt.show()

    return filt_data, emp_vals, fx


def read_raw(subject, cond='Relax'):
    dir_name = join(get_base_dir(), 'eeg-clam-tacs-cmc', 'data', subject)
    raw = read_raw_brainvision(join(dir_name, f'{cond}.vhdr'), preload=True)
    try:
        raw.drop_channels(['envelope', 'envelope_am', 'force'])
    except ValueError:
        raw.drop_channels(['envelope', 'envelope_am', 'ecg'])
    raw.set_montage(make_standard_montage('easycap-M1'), match_case=False)

    # Remove the bad channels
    bad_idx = loadmat(join(dir_name, 'exclude_idx.mat'))['exclude_idx'].squeeze() - 1
    bad_chs = [raw.ch_names[idx] for idx in bad_idx]
    raw.drop_channels(bad_chs)

    # Add the subject info
    raw.info['subject_info'] = {'id': subject.split('_')[1], 'his_id': subject}

    return raw
