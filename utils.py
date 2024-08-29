import sys
from os import makedirs
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.colors import LinearSegmentedColormap
from mne.channels import make_standard_montage
from mne.io import read_raw_brainvision
from mne.time_frequency import psd_array_welch
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import inset_locator
from scipy import linalg
from scipy.fft import fft, ifft
from scipy.io import loadmat
from scipy.signal import hilbert
from yaml import safe_load

curr_dir = dirname(__file__)


def get_base_dir():
    # Load the config file
    f_name = join(curr_dir, 'config.yaml')
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
    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[0] > data.shape[1]:
        raise ValueError(
            'Data dimensions may be incorrect. Data should be channels x time.'
        )

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
    with open(join(curr_dir, 'config.yaml'), 'r') as file:
        config = safe_load(file)

    bad_idx = loadmat(join(dir_name, 'exclude_idx.mat'))['exclude_idx'].squeeze() - 1
    bad_chs = [raw.ch_names[idx] for idx in bad_idx]

    if config['bad_channels'][subject]:
        bad_chs.extend(config['bad_channels'][subject])

    raw.drop_channels(bad_chs)

    # Add the subject info
    raw.info['subject_info'] = {'id': subject.split('_')[1], 'his_id': subject}

    return raw


def get_P_TARGET(raw, l_freq, h_freq, df, n_comps=4, save=False):
    data_signal = (
        raw.copy()
        .filter(l_freq, h_freq, l_trans_bandwidth=1, h_trans_bandwidth=1)
        ._data
    )
    data_noise = (
        raw.copy()
        .filter(l_freq - df, h_freq + df, l_trans_bandwidth=1, h_trans_bandwidth=1)
        .filter(h_freq, l_freq, l_trans_bandwidth=1, h_trans_bandwidth=1)
        ._data
    )
    data_broad = (
        raw.copy().filter(1, 30, l_trans_bandwidth=1, h_trans_bandwidth=1)._data
    )

    A = np.cov(data_signal)
    B = np.cov(data_noise)
    evals, evecs = linalg.eig(A, B)
    ix = np.argsort(evals)[::-1]
    D = evecs[:, ix].T
    M = linalg.pinv(D)

    if n_comps == 'all':
        n_comps = M.shape[0]

    for ix_comp in range(n_comps):
        psd, freqs = psd_array_welch(
            D[ix_comp] @ data_broad,
            raw.info['sfreq'],
            fmin=1,
            fmax=30,
            n_fft=int(3 * raw.info['sfreq']),
        )
        freq_mask = np.logical_and(freqs > l_freq, freqs < h_freq)
        peak_freq = freqs[freq_mask][np.argmax(psd[freq_mask])]

        fig, ax = plt.subplots()
        ax.axvline(peak_freq, color='black', ls='--', lw=0.5)
        ax.semilogy(freqs, psd, color='#045275')
        ax.axvspan(l_freq - df, l_freq, alpha=0.1, color='#F0746E')
        ax.axvspan(h_freq, h_freq + df, alpha=0.1, color='#F0746E')
        ax.axvspan(l_freq, h_freq, alpha=0.1, color='#7CCBA2')

        axins = inset_locator.inset_axes(
            ax, width='30%', height='30%', loc='upper right'
        )
        plot_topomap(M[:, ix_comp], raw.info, axes=axins)

        ax.set_title(f'Component {ix_comp:d}', loc='left')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (dB/Hz)')

        text_str = '\n'.join(
            (
                f'$f_S={l_freq}-{h_freq}$ Hz',
                f'$f_N={l_freq-df}-{h_freq+df}$ Hz',
            )
        )
        props = dict(facecolor='none', edgecolor='black')

        ax_pos = ax.get_position()
        fig.text(
            ax_pos.x1 - 0.007,
            1 - ax_pos.y0,
            text_str,
            horizontalalignment='right',
            verticalalignment='bottom',
            bbox=props,
            transform=fig.transFigure,
        )
        ax.text(
            peak_freq + 0.2,
            0.94,
            f'$PF={peak_freq:.2f}$ Hz',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.get_xaxis_transform(),
            color='black',
        )

        folder_path = join(dirname(__file__), 'img', f'{l_freq}-{h_freq}Hz')
        makedirs(folder_path, exist_ok=True)

        if save:
            plt.savefig(
                join(
                    folder_path,
                    f'ssd-{raw.info["subject_info"]["his_id"]}-comp_{ix_comp}.png',
                ),
                dpi=300,
            )
            plt.close()

        else:
            plt.show()

    return M, D, freqs, psd, peak_freq
