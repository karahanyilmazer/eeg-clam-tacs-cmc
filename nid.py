import numpy as np
import mne
from scipy import linalg
from sklearn.decomposition import FastICA
from mne.viz import plot_topomap
from mne.time_frequency import psd_array_welch
import matplotlib.pyplot as plt
from scipy.io import loadmat

def make_target(data, forward_model, COV=None):
    n_chs = len(forward_model)
    if data.ndim == 3:
        data = data[:,:n_chs]
        if COV is None:
            COV = np.cov(np.real(np.concatenate(data,axis=-1)))
        COVinv = linalg.pinv(COV)
        w = ((COVinv@forward_model[:,None])).squeeze()/(forward_model[None,:]@COVinv@forward_model[:,None])
        return np.array([w@ep for ep in data]).squeeze()
    else:
        data = data[:n_chs]
        if COV is None:
            COV = np.cov(np.real(data))
        COVinv = linalg.pinv(COV)
        w = ((COVinv@forward_model[:,None])).squeeze()/(forward_model[None,:]@COVinv@forward_model[:,None])
        return (w@data).squeeze()

def ssd(raw_target, raw_broad):
    data_target = raw_target.get_data()
    data_broad = raw_broad.get_data()
    A = np.cov(data_target)
    B = np.cov(data_broad)
    evals, evecs = linalg.eig(A,B)
    ix = np.argsort(evals)[::-1]
    D = evecs[:, ix].T
    M = linalg.pinv(D)
    return D @ data_target, D, M

def nid(raw):
    raw_alpha = raw.copy().filter(8,14)
    raw_beta = raw.copy().filter(15,25)
    raw_broad = raw.copy().filter(1,30)
    n_chs = len(raw_alpha.ch_names)
    
    # First do SSD on alpha and beta
    X_alpha, D_alpha, M_alpha = ssd(raw_alpha, raw_broad)
    X_beta, D_beta, M_beta = ssd(raw_beta, raw_broad)
    
    # Stack sources
    X_stacked = np.vstack([X_alpha, X_beta])
    
    # Do ICA on stacked sources
    ica = FastICA()
    ica.fit(X_stacked.T)
    M_ica = ica.mixing_
    
    # Project mixing matrix
    M_proj = M_alpha @ M_ica[:n_chs, :]
    n_src = M_proj.shape[1]
    
    # For each source, plot forward model and use beamformer to reconstruct source and plot its PSD
    for ix in range(n_src):
        forward = M_proj[:, ix]
        source = make_target(raw_broad.get_data(), forward)
        plot_topomap(forward, raw.info, sensors=True, show=False)
        plt.figure()
        psd,freqs = psd_array_welch(source, raw.info['sfreq'], fmin=1, fmax=30, n_fft=int(3*raw.info['sfreq']))
        plt.semilogy(freqs, psd)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')
        plt.title('Component {:d}/{:d}'.format(int(ix+1), n_src))
        plt.show()
        
raw = mne.io.read_raw_brainvision('P8_FF\\relax_no_stim.vhdr', preload=True)
ix_bads = loadmat('P8_FF\\exclude_idx.mat')['exclude_idx'].flatten()-1
raw.pick_channels(raw.ch_names[:64])
raw.drop_channels(np.array(raw.ch_names)[ix_bads])
raw.resample(200)
raw.set_montage('easycap-M1', match_case=False)
nid(raw)