'''To compare the result with the DM_phase algorithm.
   The code is mostly dupllicated with danielemichilli/DM_phase
'''

import numpy as np

# To run the parallel computing on the Scinet clusters
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import glob, math, os, sys
import astropy.units as u
import matplotlib.pyplot as plt
#matplotlib.use('agg')
#%matplotlib inline
from matplotlib import cm
import numpy.ma as ma
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import scipy.fftpack
from scipy.fftpack import fft, ifft, fftshift
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from itertools import cycle
from matplotlib.widgets import Cursor, SpanSelector, Button
import scipy.signal
import time
import argparse

parser=argparse.ArgumentParser(description='starting the process')
parser.add_argument('-data_path','--data-path', type=str, help='the path of the input data in the shape of (Number of pulses, frequency, time)',required=False)
parser.add_argument('-save_path','--save-path', type=str, help='the path of saving the test result in the npy file with a shape of (dms, dm_stds)',required=False)
args = parser.parse_args()

data_path = args.data_path
save_path = args.save_path

#define plotting parameters
plt.rcParams['toolbar'] = 'None'
plt.rcParams['keymap.yscale'] = 'Y'
colormap_list = cycle(['YlOrBr_r', 'viridis', 'Greys'])
colormap = next(colormap_list)

def _get_Spect(waterfall):
    """
    Get the coherent spectrum of the waterfall.
    """

    FT = fft(waterfall)
    amp = np.abs(FT)
    amp[amp == 0] = 1
    spect = np.sum(FT / amp, axis=0)
    return spect

def _get_Pow(waterfall):
    """
    Get the coherent power of the waterfall.
    """

    spect = _get_Spect(waterfall)
    Pow = np.abs(spect)**2
    return Pow

def get_DM(waterfall, DM_list, t_res, f_channels, ref_freq="top",
    manual_cutoff=False, manual_bandwidth=False, diagnostic_plots=True,
    fname="", no_plots=False):
    """
    Brute-force search of the Dispersion Measure of a waterfall numpy matrix.
    The algorithm uses phase information and is robust to interference and unusual burst shapes.
    Parameters
    ----------
    waterfall : ndarray
        2D array with shape (frequency channels, phase bins)
    DM_list : list
        List of Dispersion Measure values to search (pc/cc).
    t_res : float
        Time resolution of each phase bin (s).
    f_channels : list
        Central frequency of each channel, from low to high (MHz).
    ref_freq : str, optional. Default = "top"
        Use either the "top", "center" or "bottom" of the band as
        reference frequency for dedispersion.
    manual_cutoff : bool, optional. Default = False
        If False, the power spectrum cutoff is automatically selected.
    manual_bandwidth : bool, optional. Default = False
        If False, use the full frequency bandwidth.
    diagnostic_plots : bool, optional. Default = True
        Stores the diagnostic plots "Waterfall_5sig.pdf" and "DM_Search.pdf"
    fname : str, optional. Default = ""
        Filename used as a prefix for the diagnostic plots.
    Returns
    -------
    DM : float
        Best value of Dispersion Measure (pc/cc).
    DM_std :
        Standard deviation of the Dispersion Measure (pc/cc)
    """
    if manual_bandwidth:
        low_ch_idx, up_ch_idx = _get_frequency_range_manual(waterfall,
            f_channels)
    else:
        low_ch_idx = 0
        up_ch_idx = waterfall.shape[0]

    waterfall = waterfall[low_ch_idx:up_ch_idx,...]
    f_channels = f_channels[low_ch_idx:up_ch_idx]

    nchan = waterfall.shape[0]
    nbin = int(waterfall.shape[1] / 2)
    Pow_list = np.zeros([nbin, DM_list.size])
    for i, DM in enumerate(DM_list):
        waterfall_dedisp = _dedisperse_waterfall(waterfall, DM, f_channels, t_res, ref_freq=ref_freq)
        Pow = _get_Pow(waterfall_dedisp)
        Pow_list[:, i] = Pow[: nbin]

    v = np.arange(0, nbin)
    dPow_list = Pow_list * v[:, np.newaxis]**2

    Mean     = nchan               # Base on Gamma(2,)
    STD      = nchan / np.sqrt(2)  # Base on Gamma(2,)
    if manual_cutoff: low_idx, up_idx, phase_lim = _get_f_threshold_manual(Pow_list, dPow_list, waterfall, DM_list, f_channels, t_res, ref_freq=ref_freq)
    else:
        low_idx, up_idx = _get_f_threshold(Pow_list, Mean, STD)
        phase_lim = None

    DM, DM_std = _DM_calculation(waterfall, Pow_list, dPow_list, low_idx, up_idx, f_channels, t_res, DM_list, no_plots=no_plots, fname=fname, phase_lim=phase_lim)
    return DM, DM_std
  
def _DM_calculation(waterfall, Pow_list, dPow_list, low_idx, up_idx, f_channels, t_res, DM_list, no_plots=False, fname="", phase_lim=None):
    """
    Calculate the best DM value.
    """

    DM_curve = dPow_list[low_idx : up_idx].sum(axis=0)

#    print('DM_curve',DM_curve)

    fact_idx = up_idx - low_idx
    Max   = DM_curve.max()
    nchan = len(f_channels)
    Mean  = nchan              # Base on Gamma(2,)
    STD   = Mean / np.sqrt(2)  # Base on Gamma(2,)
    m_fact = np.sum(np.arange(low_idx, up_idx)**2)
    s_fact = np.sum(np.arange(low_idx, up_idx)**4)**0.5
    dMean = Mean * m_fact
    dSTD  = STD  * s_fact
    SN    = (Max - dMean) / dSTD

    Peak  = DM_curve.argmax()
    Range = np.arange(Peak - 5, Peak + 5) # for sim-1,2,4-9
#    Range = np.arange(Peak - 15, Peak + 15) # for sim-3


#    print('Range',Range)
    y = DM_curve[Range]
    x = DM_list[Range]
    Returns_Poly = _Poly_Max(x, y, dSTD)

    if not no_plots:
        _plot_Power(Pow_list, low_idx, up_idx, DM_list, DM_curve, Range, Returns_Poly, x, y, SN, t_res, fname=fname)
        _plot_waterfall(Returns_Poly, waterfall, t_res, f_channels, fact_idx, fname=fname, Win=phase_lim)

    DM = Returns_Poly[0]
    DM_std = Returns_Poly[1]
    return DM, DM_std

def _dedisperse_waterfall(wfall, DM, freq, dt, ref_freq="top"):
    """
    Dedisperse a wfall matrix to DM.
    """

    k_DM = 1. / 2.41e-4
    dedisp = np.zeros_like(wfall)

    # pick reference frequency for dedispersion
    if ref_freq == "top":
        reference_frequency = freq[-1]
    elif ref_freq == "center":
        center_idx = len(freq) // 2
        reference_frequency = freq[center_idx]
    elif ref_freq == "bottom":
        reference_frequency = freq[0]
    else:
        print("`ref_freq` not recognized, using 'top'")
        reference_frequency = freq[-1]

    shift = (k_DM * DM * (reference_frequency**-2 - freq**-2) / dt).round().astype(int)
    for i,ts in enumerate(wfall):
        dedisp[i] = np.roll(ts, shift[i])
    return dedisp

def _get_f_threshold(Pow_list, MEAN, STD):
    """
    Get the Fourier frequency cutoff.
    """

    s   = np.max(Pow_list, axis=1)
    SN  = (s - MEAN) / STD
    Kern = np.round( _get_Window(SN)/2 ).astype(int)
    if Kern < 5: Kern = 5
    return 0, Kern

def _get_Window(Pro):
    """
    ACF Windowing
    """

    arr = scipy.signal.detrend(Pro)
    X = np.correlate(arr, arr, "same")
    n = X.argmax()
    W = np.max(np.diff(np.where(X < 0)))
    return W

def _Poly_Max(x, y, Err):
    """
    Polynomial fit
    """
    n = np.linalg.matrix_rank(np.vander(y))
    p = np.polyfit(x, y, n)
    Fac = np.std(y) / Err

    dp      = np.polyder(p)
    ddp     = np.polyder(dp)
    cands   = np.roots(dp)
    r_cands = np.polyval(ddp, cands)
    first_cut = cands[(cands.imag==0) & (cands.real>=min(x)) & (cands.real<=max(x)) & (r_cands<0)]
    if first_cut.size > 0:
        Value     = np.polyval(p, first_cut)
        Best      = first_cut[Value.argmax()]
        delta_x   = np.sqrt(np.abs(2 * Err / np.polyval(ddp, Best)))
    else:
        Best    = 0.
        delta_x = 0.

    return float(np.real(Best)), delta_x, p , Fac

def _plot_Power(DM_Map, low_idx, up_idx, X, Y, Range, Returns_Poly, x, y, SN, t_res, fname=""):
    """
    Diagnostic plot of Coherent Power vs Dispersion Measure
    """

    fig = plt.figure(figsize=(6, 8.5), facecolor='k')
    fig.subplots_adjust(left=0.1, bottom=0.05, right=0.99, top=0.88)
    gs = gridspec.GridSpec(3, 1, hspace=0, height_ratios=[3, 1, 9])
    ax_prof = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_prof)
    ax_map = fig.add_subplot(gs[2], sharex=ax_prof)

    Title = '{0:}\n\
        Best DM = {1:.3f} $\pm$ {2:.3f}\n\
        S/N = {3:.1f}'.format(fname, Returns_Poly[0], Returns_Poly[1], SN)
    fig.suptitle(Title, color='w', linespacing=1.5)

    # Profile
    ax_prof.plot(X, Y, 'w-', linewidth=3, clip_on=False)
    ax_prof.plot(X[Range], np.polyval(Returns_Poly[2], X[Range]), color='orange', linewidth=3, zorder=2, clip_on=False)
    ax_prof.set_xlim([X.min(), X.max()])
    ax_prof.set_ylim([Y.min(), Y.max()])
    ax_prof.axis('off')
    ax_prof.ticklabel_format(useOffset=False)

    # Residuals
    Res = y - np.polyval(Returns_Poly[2], x)
    Res -= Res.min()
    Res /= Res.max()
    ax_res.plot(x, Res, 'xw', linewidth=2, clip_on=False)
    ax_res.set_ylim([np.min(Res) - np.std(Res) / 2, np.max(Res) + np.std(Res) / 2])
    ax_res.set_ylabel('$\Delta$')
    ax_res.tick_params(axis='both', colors='w', labelbottom='off', labelleft='off', direction='in', left='off', top='on')
    ax_res.yaxis.label.set_color('w')
    try: ax_res.set_facecolor('k')
    except AttributeError: ax_res.set_axis_bgcolor('k')
    ax_res.ticklabel_format(useOffset=False)

    # Power vs DM map
    FT_len = DM_Map.shape[0]
    indx2Ang = 1. / (2 * FT_len * t_res * 1000)
    extent = [np.min(X), np.max(X), low_idx * indx2Ang, up_idx * indx2Ang]
    ax_map.imshow(DM_Map[low_idx : up_idx], origin='lower', aspect='auto', cmap=colormap, extent=extent, interpolation='nearest')
    ax_map.tick_params(axis='both', colors='w', direction='in', right='on', top='on')
    ax_map.xaxis.label.set_color('w')
    ax_map.yaxis.label.set_color('w')
    ax_map.set_xlabel('DM (pc cm$^{-3}$)')
    ax_map.set_ylabel('Fluctuation Frequency (ms$^{-1}$)')  #From p142 in handbook, also see Camilo et al. (1996)
    ax_map.ticklabel_format(useOffset=False)
   try: fig.align_ylabels([ax_map, ax_res])  #Recently added feature
    except AttributeError:
        ax_map.yaxis.set_label_coords(-0.07, 0.5)
        ax_res.yaxis.set_label_coords(-0.07, 0.5)

    if fname != "": fname += "_"
    fig.savefig(fname + "DM_Search.pdf", facecolor='k', edgecolor='k')
    return

def _plot_waterfall(Returns_Poly, waterfall, dt, f, Cut_off, fname="", Win=None):
    """
    Plot the waterfall at the best Dispersion Measure and at close values for comparison.
    """

    fig = plt.figure(figsize=(8.5, 6), facecolor='k')
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.8)
    grid = gridspec.GridSpec(1, 3, wspace=0.1)

    Title='{0:}\n\
        Best DM = {1:.3f} $\pm$ {2:.3f}'.format(fname, Returns_Poly[0], Returns_Poly[1])
    plt.suptitle(Title, color='w', linespacing=1.5)

    DMs = Returns_Poly[0] + 5 * Returns_Poly[1] * np.array([-1, 0, 1])  # DMs +- 5 sigmas away
    for j, dm in enumerate(DMs):
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[j], height_ratios=[1, 4], hspace=0)
        ax_prof = fig.add_subplot(gs[0])
        ax_wfall = fig.add_subplot(gs[1], sharex=ax_prof)
        try: ax_wfall.set_facecolor('k')
        except AttributeError: ax_wfall.set_axis_bgcolor('k')

        wfall = _dedisperse_waterfall(waterfall, dm, f, dt)
        prof = wfall.sum(axis=0)

        # Find the time range around the pulse
        if (j == 0) and (Win is None):
            W = _get_Window(prof)
            Spect = _get_Spect(wfall)
            Filter = np.ones_like(Spect)
            Filter[Cut_off : -Cut_off] = 0
            Spike = np.real(ifft(Spect * Filter))
           Spike[0] = 0
            Win = _check_W(Spike, W)

        # Profile
        T = dt * (Win[1] - Win[0]) * 1000
        x = np.linspace(0, T, Win[1] - Win[0])
        y = prof[Win[0] : Win[1]]
        ax_prof.plot(x, y, 'w', linewidth=0.5, clip_on=False)
        ax_prof.axis('off')
        ax_prof.set_title('{0:.3f}'.format(dm), color='w')

        # Waterfall
        bw = f[-1] - f[0]
        im = wfall[:, Win[0] : Win[1]]
#        extent = [0, T, f[0], f[-1]]
        print('f',f)
        extent = [0, T, f[0].value, f[-1].value]
        MAX_DS = wfall.max()
        MIN_DS = wfall.mean() - wfall.std()
        ax_wfall.imshow(im, origin='lower', aspect='auto', cmap=colormap, extent=extent, interpolation='nearest', vmin=MIN_DS, vmax=MAX_DS)
#        ax_wfall.imshow(im, origin='lower', aspect='auto', cmap=colormap, interpolation='nearest', vmin=MIN_DS, vmax=MAX_DS)


        ax_wfall.tick_params(axis='both', colors='w', direction='in', right='on', top='on')
        if j == 0: ax_wfall.set_ylabel('Frequency (MHz)')
        if j == 1: ax_wfall.set_xlabel('Time (ms)')
        if j > 0: ax_wfall.tick_params(axis='both', labelleft='off')
        ax_wfall.yaxis.label.set_color('w')
        ax_wfall.xaxis.label.set_color('w')

    if fname != "": fname += "_"
    fig.savefig(fname + "Waterfall_5sig.pdf", facecolor='k', edgecolor='k')
    return

def _check_W(Pro, W):
    """
    Check whether the veiwing window will be in the index range.
    """

    SM = np.convolve(Pro, np.ones(W), 'same')
   Peak = np.mean(np.where(SM == max(SM)))
    Max = np.where(Pro == np.max(Pro))
    if (Peak - Max)**2 > W**2:
        W += np.abs(Peak - Max) / 2
        Peak = (Peak + Max) / 2
    Start = np.int(Peak - np.round(1.25 * W))
    End = np.int(Peak + np.round(1.25 * W))
    if Start < 0: Start=0
    if End > Pro.size - 1: End = Pro.size - 1
    return Start,End

def process_single(waterfall):

#    t0=time.time()
#    I=np.load('/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/sim1_individual_test.npz', allow_pickle=True)['I_files'][0]

#    waterfall = I.copy()
#    DM_list = np.linspace(-2,2,51,endpoint=True) # for sim-1,2,4-9
    DM_list = np.linspace(-2,2,1001,endpoint=True) # for sim-1
    t_res = 0.00032768
    f_channels = 550*u.MHz+200*u.MHz*np.arange(2048)/2048

    dm, dm_std = get_DM(waterfall, DM_list, t_res, f_channels, ref_freq="top",
        manual_cutoff=False, manual_bandwidth=False, diagnostic_plots=False,
        fname="", no_plots=False)
    print('dm, dm_std',dm, dm_std)
    return dm, dm_std

def process_multi(mpi_elements, waterfalls):

    dms, dm_stds = [], []
#    print('waterfalls.shape',waterfalls.shape)
    for mpi_element in mpi_elements:
#        print('mpi_element', mpi_element)
        waterfall = waterfalls[mpi_element,:,:]
#        print('waterfall.shape',waterfall.shape)
        dm, dm_std = process_single(waterfall)
        dms.append(dm)
        dm_stds.append(dm_std)
    return dms, dm_stds
def multi_cores(waterfalls, save_path):


    # the testing times
    N = 100
    mpi_elements = np.array_split(np.arange(N), size)
    mpi_elements = comm.scatter(mpi_elements, root=0)

    # do calculation in each rank
    dms, dm_stds = process_multi(mpi_elements, waterfalls)

    ''' gather all results to rank 0'''
    dms = comm.gather(dms,root=0)
    dm_stds = comm.gather(dm_stds,root=0)

    if rank==0:
        dms = np.concatenate(dms)
        dm_stds = np.concatenate(dm_stds)
        # save the result in rank=0
        np.savez(save_path, dms=dms, dm_stds=dm_stds)


def main():


    waterfalls = np.load(data_path, allow_pickle=True)['I_files']
    multi_cores(waterfalls, save_path)


if __name__ == '__main__':
    main()
