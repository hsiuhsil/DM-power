import numpy as np

# To run the parallel computing on the Scinet clusters
#import mpi4py.rc
#mpi4py.rc.threads = False
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()

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
import argparse

parser=argparse.ArgumentParser(description='starting the process')
parser.add_argument('-bw','--bandwidth', type=float, default=200, help='the bandwidth in MHz',required=True)
parser.add_argument('-f0','--bottom-band', type=float, default=550, help='the bottom band in MHz',required=True)
parser.add_argument('-nchan','--nchan', type=int, default=2048, help='the number of channels',required=True)
parser.add_argument('-dt','--dt', type=float, default=0.00032768, help='the time resolution of the data in second',required=True)
parser.add_argument('-dm_start','--dm-start', default=-2, type=float, help='the starting DM value for the optimization in pc/cm**3',required=True)
parser.add_argument('-dm_end','--dm-end', default=2, type=float, help='the ending DM value for the optimization in pc/cm**3',required=True)
parser.add_argument('-dm_steps','--dm-steps', default=51, type=int, help='the numer of DM steps for the optimization',required=True)
parser.add_argument('-trials','--trials', default=1000, type=int, help='the numer of trials in the bootstrap tests',required=True)
parser.add_argument('-intensity_file','--intensity-file', type=str, help='the path of the intensity in the npy file with a shape of (frequency, time)',required=True)
args = parser.parse_args()


'''define the systematic parameters'''
# note: For the GMRT observations of R3, BW=200 MHz, f0=550 MHz, nchan=2048, dt=0.00032768 sec, dm_start=-2, dm_end=2, dm_steps=51, trials=1000.
parser=argparse.ArgumentParser(description='starting the process')
BW = (args.bandwidth)*u.MHz
f0 = (args.bottom_band)*u.MHz
nchan = args.nchan
chan_bw = BW/nchan # the bandwidth of each channel
dt = (args.dt)*u.s #0.32768*u.ms
f_arr = f0+BW*np.arange(nchan)/nchan

# define the dm series
dm_start = args.dm_start
dm_end = args.dm_end
dm_steps = args.dm_steps
dm_series = np.linspace(dm_start, dm_end, dm_steps, endpoint=True)

# define the trials:
trials = args.trials # 0 without bootstraping, default for 1000

# defint the intensity file
intensity_file = args.intensity_file

def ddtime(fref,freq,DM):
    # the function to calculate the dispersion time delay 
    # fref: the reference frequency ("Top freq", central freq, bottom freq)
    # freq: the value of the specific channel
    # DM: Dispersion Measure
    # return the time in second
    D = (1/2.41e-4) * u.s * u.MHz**2 * u.cm**3 / u.pc
    return D*DM*(freq**-2-fref**-2)

def spec_shiftDM(data, DM, freqArr):
    # keep the data without rebinning (so that have a high DM resolution)
    #data should be in the shape of (ntime,nchan)
    # DM input is a value without unit
    #freqArr describes the frequency range of the data
    
    # data means the intensity spectrum, in the shape of (time, freq)
    # freqArr, the aequence of frequency array 
    
    data = data.T # to be in the shape of (nchan, ntime)
    fref = freqArr.max() # top of the band to be the reference freq
    dataNew = np.zeros(data.shape) # creating a new matrix to store the intensity with DM shift
    DM = DM*u.pc / u.cm**3 #give DM with the unit
    
    for i in range(data.shape[0]): # to do incoh-dd for each of the freq channels
        
        # below, we are using astropy to eventually remove the unit
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            # decide the how many chunks that the spectrum will be shifted in that freq
            roll = ddtime(fref,freqArr[i],DM)/dt
            
            if True:
                shift = roll.decompose().value
                fftfreq = scipy.fftpack.fftfreq(len(data[i]))
                fourier_shift  = np.exp(1j*2*np.pi*(shift*fftfreq))
                dataNew[i] = np.fft.ifft(np.fft.fft(data[i])*fourier_shift)
            
            else: # to get the roll amount in an interger
                roll = int(roll.decompose().value)
                # doing the shifting
                dataNew[i] = np.roll(data[i],-roll)
    return dataNew

def rebin(matrix, xbin, ybin):
    # rebinning for a 2D case
    # matrix is your input data
    # xbin means the factor that how many nearby pixels in the horizontal axis (matrix.shape[1]) one wants to average.
    # ybin means the factor that how many nearby pixels in the vertical axis (matrix.shape[0]) one wants to average.
    
    shape1=matrix.shape[0]//xbin
    shape2=matrix.shape[1]//ybin
    return np.nanmean(np.reshape(matrix[:shape1*xbin,:shape2*ybin],(shape1,xbin,shape2,ybin)),axis=(1,3))

def rebin1D(array, xbin):
    # the rebinning for a 1D case
    # the array is a 1D array/data
    # the xbin means the facotr that how many nearby pixels one wants to average. 
    
    shape1=len(array)//xbin
    return np.nanmean(np.reshape(array[:shape1*xbin],(shape1,xbin)),axis=(1))

def svd(data):
    # apply the singular value decomposition to the data, which is in the shape of (freq, time).
    U, s, V = np.linalg.svd(data, full_matrices=False)

    if 0 < np.abs(np.amin(U[0])):
        U[0] = -U[0]      

    if False:
        s /= s.sum()
        
    return U,s,V


def plot_svd(data, name, savefig=False):
    # plotting the SVD results.

    U,s,V=svd(data)
    
    fig, (ax0, ax1,ax2) = plt.subplots(nrows=1, ncols=3, sharex=False,
                                    figsize=(16, 8))
    fig.suptitle('SVD analysis of '+name,fontsize=24,y=1.05)
    fig.subplots_adjust(top=0.9)

    
    U=U.T
    for i in range(8):        
        ax0.plot(f_arr, U[i]-i*0.5,label='U'+str(i))
    
    ax0.set_xlabel('freq (MHz)')
    ax0.set_ylabel('Modes')
    ax0.set_title('U modes',fontsize=16)
    xarr = np.arange(8)
    xticklabels = (np.arange(8)+1).tolist()
    yarr = np.arange(8)*-0.5
    yticklabels = (np.arange(8)+1).tolist()
    
    
    ax0.set_yticks(yarr)
    ax0.set_yticklabels(yticklabels)
    ax0.grid(True)
    
    ax1.scatter(np.arange(len(s[:20])),s[:20])
    ax1.set_title('S modes',fontsize=16)
    ax1.set_xlabel('Modes')
    xarr2 = np.arange(20)
    xticklabels2 = (np.arange(20)+1).tolist()
    ax1.set_xticks(xarr2)
    ax1.set_xticklabels(xticklabels2)
    ax1.set_xlim((-1,20))
    ax1.grid(True)

    for i in range(8):
        t = dt*np.arange(len(V[i]))
        ax2.plot(t, V[i]-i*0.5,label='V'+str(i))
    
    ax2.set_xlabel('time (ms)')
    ax2.set_ylabel('Modes')
    ax2.set_title('V modes',fontsize=16)
    
    ax2.set_yticks(yarr)
    ax2.set_yticklabels(yticklabels)
    ax2.grid(True)
    
    fig.tight_layout()
    
    if savefig==True:
        filename = 'svd.png'
        plt.savefig(filename, dpi=50, bbox_inches='tight')
    
    plt.show()
    
def fft_power_arr(arr):
    # input an array, doing the fft, fftshift
    fft_arr = scipy.fftpack.fft(arr)
    fft_arr = scipy.fftpack.fftshift(fft_arr)
    # getting the power by the FFT array times its complex conjugate
    fft_arr = fft_arr*np.conj(fft_arr)
    return fft_arr # return the power

def log_rebin1D(z, x, x_bins):
    
    # z is the 1D data in linear scale, which one wanted to rebin  from linear to log scale.
    # x is the linear array (such the horizontal axis of time or freq)
    # x_bins should be the preferred log-scale of x 
    
    log_rebinned = np.zeros([x_bins.size], dtype = float)
    n_avg = np.zeros([x_bins.size])
    for i in range(len(z)):
        current_x =x[i]
        # x and y_bin_idx is the index of the bin in 'x' where k belongs.
        x_bin_idx = np.where(abs(current_x-x_bins) == min(abs(current_x-x_bins)))[0][0]
        # at this index, in the rebinned plot, add the Z value.
        log_rebinned[x_bin_idx] += z[i]
        # keep track of how many Z values we add to a given rebinned bin to avg.
        n_avg[x_bin_idx] += 1
    z = log_rebinned/n_avg
    return z

def get_power(I, dm_series, trials):
    #define the log delay-freq
    pulse_bins=int(I.shape[1]/2)
    duration = np.arange(pulse_bins)*dt.value
    # convert the original freq/time binsize into the delay time/freq binsize 
    delay_freq_bin = (1/(pulse_bins*dt)).to(u.Hz) #Hz
    delay_time_bin = (1/(nchan*chan_bw)).to(u.second)*1e9 #nanoosec

    # getting the corresponding range of the waterfall in the FFT domain
    delay_freq_range = np.linspace(-int(pulse_bins/2), int(pulse_bins/2), pulse_bins,endpoint=True) *2* delay_freq_bin
    delay_time_range = np.linspace(-int(nchan/2), int(nchan/2), nchan,endpoint=True) * delay_time_bin

    pos_freq = np.where(delay_freq_range>=0)[0][0]
    log_min = (np.log10(np.abs(delay_freq_range.value).min())//0.1)/10
    log_max = (np.log10(np.abs(delay_freq_range.value).max())//0.1+1)/10
    log_num = 10
    log_delay_freq_range = np.concatenate((np.asarray([0]),np.logspace(log_min,log_max,log_num)))

    dm_power = np.zeros((len(dm_series), trials, len(delay_freq_range.value[pos_freq:])))
    dm_power_log = np.zeros((len(dm_series), trials, len(log_delay_freq_range)))

    import time
    T0=time.time()
    # for each of the DM:
    for dm, j in zip(dm_series, np.arange(len(dm_series))):
    
        # incoh-DD at that DM
        I_shift = spec_shiftDM(I.T, dm, f_arr)
    
        on_pulse = I_shift[:,int(0.25*I.shape[1]):int(0.75*I.shape[1])]
        off_pulse = np.concatenate((I_shift[:,0:int(0.25*I.shape[1])],I_shift[:,int(0.75*I.shape[1]):]),axis=-1)
        
        # apply the SVD and get the normalization
        U_on, s_on, V_on = svd(on_pulse)
        U_off, s_off, V_off = svd(off_pulse)

        # getting the complex conjugate of the U from the on-pulse
        w0 = np.linalg.pinv(U_on) 
    
        # getting U0
        U0 = w0[0]
        waterfall_on = on_pulse*(U0[:, np.newaxis])
        waterfall_off = off_pulse*(U0[:, np.newaxis])

        '''applying bootstrap'''  
        for trail in range(trials):
            s = np.random.choice(np.arange(nchan),size=nchan)
        
            pulse_svd = np.sum(waterfall_on[s,:],axis=0)
            noise_svd = np.sum(waterfall_off[s,:],axis=0)            

            if True: 
                pulse_svd = np.abs(pulse_svd)
                noise_svd = np.abs(noise_svd)

            fft_v0_pulse = fft_power_arr(pulse_svd)
            fft_v0_noise = fft_power_arr(noise_svd)
    
            '''log bins of the power spectrum'''
            # getting the on minus off (to remove the noise power)
            fft_v0_pulse_subtract = fft_v0_pulse[pos_freq:] - fft_v0_noise[pos_freq:]
            # getting the power-spectrum in the log-rebin1D 
            log_fft_v0_pulse_subtract = log_rebin1D(fft_v0_pulse_subtract, delay_freq_range.value[pos_freq:], log_delay_freq_range)
 
            #print(len(log_fft_v0_pulse_subtract))
            dm_power[j,trail] = fft_v0_pulse_subtract
            dm_power_log[j,trail]=log_fft_v0_pulse_subtract

        if j%10==0:
            print('finished '+str(j+1)+'/ '+str(len(dm_series))+' DM steps')
        
    return delay_freq_range, log_delay_freq_range, dm_power, dm_power_log


def gaus(x,a,x0,sigma, offset):
#    a,x0,sigma, offset = pars[0], pars[1], pars[2], pars[3]
    return a*exp(-(x-x0)**2/(2*sigma**2)) + offset


def fit_log_dm_width(log_delay_freq_range, trials, dm_power_log):
    fit_width = np.zeros((trials,len(log_delay_freq_range)))
    fit_dm = np.zeros((trials,len(log_delay_freq_range)))

    for f in range(len(log_delay_freq_range)):
        for trail in range(trials):
            try:
                data = dm_power_log[:,trail,f]
                pars_init=[np.amax(data),dm_series[np.argmax(data)],0.2,0]
                popt,pcov = curve_fit(gaus,dm_series,data,
                              p0=pars_init,
                               bounds=((0, -10, 0, -np.inf), 
                                          (np.inf, 10, np.inf, np.inf)))
    
                fit_width[trail,f] = popt[2]
            except (RuntimeError, ValueError):
                fit_width[trail,f] = np.nan
    
    
        median_width = np.nanmedian(fit_width[:,f])

        if False:
            plt.figure(figsize=(8,6))
            plt.title('the delay freq at '+"%.2f"%log_delay_freq_range[f]+' Hz')
            h = fit_width[:,f]
            plt.hist(h[~np.isnan(h)])
            plt.xlabel('the individual fitting width')
            plt.ylabel('counts')
            plt.show()

        '''fitting with a fixed width'''
        eps=1e-5
        for trail in range(trials):
            try:
                data = dm_power_log[:,trail,f]
                pars_init=[np.amax(data),dm_series[np.argmax(data)],median_width,0]
                popt,pcov = curve_fit(gaus,dm_series,data,
                              p0=pars_init,
                              bounds=((0, -10, median_width-eps, -np.inf), 
                                      (np.inf, 10, median_width+eps, np.inf)))
                
                fit_dm[trail,f] = popt[1]
            
                model = gaus(dm_series, popt[0], popt[1], popt[2], popt[3])
        
            except (RuntimeError, ValueError):
                fit_dm[trail,f] = np.nan
        
        if False:
            plt.figure(figsize=(8,6))
            plt.title('the delay freq at '+"%.4f"%log_delay_freq_range[f]+' Hz'+'\n'+'mean: '+"%.3f"%np.nanmean(fit_dm[:,f])+', std: '+"%.3f"%np.nanstd(fit_dm[:,f]))
            h = fit_dm[:,f]
            plt.hist(h[~np.isnan(h)],bins=20)
            plt.xlabel('the individual fitting DM')
            plt.ylabel('counts')
            plt.show()
        
    return fit_width, fit_dm


def plot_dm_err(fit_dm, log_delay_freq_range):
    '''plotting the DM v.s. delay freq'''

    m = np.nanmean(fit_dm,axis=0)
    s = np.nanstd(fit_dm,axis=0)
    w = (s**-2)
    
    opt_dm = np.nansum(m*w)/np.nansum(w)
    opt_dm_err = 1/np.sqrt(np.nansum(w))

    if False:
        title = ' The optimized DM: '+"%.4f"%(opt_dm) +\
                ' DM-error of '+"%.4f"%(opt_dm_err)

        fig, ((ax1,ax2)) = plt.subplots(2,1, figsize = (12,8))
        fig.subplots_adjust(wspace=0.25,hspace=0.2)
        fig.suptitle(title)

        ax1.scatter(log_delay_freq_range,m)
        ax1.axes.axhline(opt_dm, linestyle='--',color='r')
        ax1.set_xscale('log')
        ax1.set_ylabel('The optimized DM')
        ax1.tick_params('both', length=20, width=2, which='major')
        ax1.tick_params('both', length=8, width=1, which='minor')
        ax1.set_xlim(1e0,1e4)
        ax1.set_ylim(-0.5,0.5)
        ax1.grid(True)

        ax2.scatter(log_delay_freq_range,s)
        ax2.axes.axhline(opt_dm_err, linestyle='-.',color='r')
        ax2.set_xlabel('Delay frequency (Hz)')
        ax2.set_ylabel('The DM error')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(1e0,1e4)
        ax2.set_ylim(5e-4,1e2)
        ax2.tick_params('both', length=20, width=2, which='major')
        ax2.tick_params('both', length=8, width=1, which='minor')
        ax2.grid(True)
        plt.show()
        if False:
            plt.savefig(path+str(burst_no)+'_optDM.png', dpi=50, bbox_inches='tight')
    
    
    return opt_dm, opt_dm_err

def main():

    '''Performing the DM optimization on the intensity file (in the shape of frequency, time).
       The outputs are the optimized DM value and the uncertainty.  
    '''

    # loading the intensity_file, which is in the shape of (frequency, time) and has been dedispersed to an initial DM. 
    I=np.load(intensity_file)

    # convert the intensity into power spectrum with DM steps and N trials of bootstrap
    delay_freq_range, log_delay_freq_range, dm_power, dm_power_log = get_power(I, dm_series, trials)

    # getting the fit DM of each freq and each trail with a fixed Gaussian width.
    fit_width, fit_dm = fit_log_dm_width(log_delay_freq_range, trials, dm_power_log)
    opt_dm, opt_dm_err = plot_dm_err(fit_dm, log_delay_freq_range)

    if True: # see the publication for the rescaling factor of sqrt(5.09)
        opt_dm_err*=np.sqrt(5.09)

    print('The optimized DM: '+str(np.round(opt_dm,4))+' pc/cm**3')
    print('The optimized DM uncertainty: '+str(np.round(opt_dm_err,4))+' pc/cm**3')
    
    return opt_dm, opt_dm_err


if __name__ == '__main__':
    main()

