# DM_power
## The purpose of this algorithm 
The algorithm to optimize dispersion measure (DM) for micro-structures of single pulses, such as Fast Radio Bursts (FRBs) or Pulsars, by maximizing the sub-structure in the power spectrum 
* The link to [the preprint PDF](https://arxiv.org/pdf/2208.13677.pdf) and [the NASA ADS page](https://ui.adsabs.harvard.edu/abs/2022arXiv220813677L/abstract).


## The usage of the algorithm

### The required intensity file

* A `npy` file with a shape of (frequency, time), and the 0-index row corresponds to the lowest frequency channel.
* The data should be dedispersed to an initial DM value. The algorithm will determine the optimized DM value by given parameters.
* The radio-frequency-interference (RFI) should be removed already.
* A sufficient duration of the intensity file is required. 
    * The 0-25 and 75-100 % of the duration are for the off-pulse region.
    * The 25-75 % of the duration is for the on-pulse region.
        * The duration of the on-pulse should cover all of the DM ranges for the optimization.  

### The required parameters:

* `-bw`: the bandwidth in MHz 
* `-f0`: the bottom of the band in MHz
* `-nchan`: the number of channels
* `-dt`: the timing resolution in seconds of the data 
* `-dm_start`: the starting DM value in pc cm<sup>-3</sup> 
* `-dm_end`: the ending DM value in pc cm<sup>-3</sup> 
* `-dm_steps`: The number of DM steps between the starting and ending DM values 
* `-trials`: the number of bootstrap tests
* `-intensity_file`: the path of the intensity data


### An example of the usage

#### Generating a simulated Gaussian profile
- `mpirun -np 100 python DM_power.py -bw 200 -f0 550 -nchan 2048 -dt 0.00032768 -dm_start -2 -dm_end 2 -dm_steps 51 -trials 100 -rescaled False  -intensity_sim True -sim_width 64 -sim_mu 0 -sim_sigma 0.25 -save_path "/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/sim1_individual_test.npz"`
    - We use the data with 2048 frequency channels from 550 to 750 MHz with a timing resolution of 0.00032768 seconds.
    - The range of the optimized DM value, different from the initially dedispersed data, is from -2 to 2 (pc cm<sup>-3</sup>). 
    - We set 100 trials for the bootstrap tests.
    - We simulate a Gaussian profile with parameters of width, the average and the variance. 
    - Finally, save the simulated profile of 100 bootstrapping Gaussian profiles.

#### Measuring the DM and the unvertainty of the simulated files.
- `mpirun -np 100 python DM_power.py -bw 200 -f0 550 -nchan 2048 -dt 0.00032768 -dm_start -2 -dm_end 2 -dm_steps 51 -trials 100 -rescaled False  -intensity_bootstrap True -intensity_file "/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/sim1_single_profile.npy" -save_path "/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/sim1_bootstrap_test.npz"`
    - We use the data with 2048 frequency channels from 550 to 750 MHz with a timing resolution of 0.00032768 seconds.
    - The range of the optimized DM value, different from the initially dedispersed data, is from -2 to 2 (pc cm<sup>-3</sup>). 
    - We set 100 trials for the bootstrap tests.
    - `-intensity_bootstrap True` means we are using bootstrap to probe the uncertainty.
    - `-intensity_file`, the path of the intensity file (note: It should be dedispersed to an initial DM value and the RFI should be masked beforehand)
    - Finally, save the results in a npz file.

#### Measuring the DM and the uncertainty of a R3 burst.
- `mpirun -np 100 python DM_power.py -bw 200 -f0 550 -nchan 2048 -dt 0.00032768 -dm_start -2 -dm_end 2 -dm_steps 51 -trials 100 -rescaled False  -intensity_bootstrap True -intensity_file "/scratch/p/pen/hsiuhsil/DM_power/noiseamp_files/burst_11_noiseamp_0.npy" -save_path "/scratch/p/pen/hsiuhsil/DM_power/noiseamp_files/burst_11_noiseamp_0_bootstrap_test.npz"`
    - We use the data with 2048 frequency channels from 550 to 750 MHz with a timing resolution of 0.00032768 seconds.
    - The range of the optimized DM value, different from the initially dedispersed data, is from -2 to 2 (pc cm<sup>-3</sup>). 
    - We set 100 trials for the bootstrap tests.
    - `-intensity_bootstrap True` means we are using bootstrap to probe the uncertainty.
    - `-intensity_file`, the path of the intensity file (note: It should be dedispersed to an initial DM value and the RFI should be masked beforehand)
    - Finally, save the results in a npz file.

#### Measuring the DM and the uncertainty of a PSR B0329+54 pulse
- `mpirun -np 100 python DM_power.py -bw 200 -f0 550 -nchan 2048 -dt 0.00008192 -dm_start -2 -dm_end 2 -dm_steps 51 -trials 100 -rescaled False  -intensity_bootstrap True -intensity_file "/scratch/p/pen/hsiuhsil/DM_power/noiseamp_files/burst_100_noiseamp_0.npy" -save_path "/scratch/p/pen/hsiuhsil/DM_power/noiseamp_files/burst_100_noiseamp_0_bootstrap_test.npz"`
    - We use the data with 2048 frequency channels from 550 to 750 MHz with a timing resolution of 0.00008192 seconds.
    - The range of the optimized DM value, different from the initially dedispersed data, is from -2 to 2 (pc cm<sup>-3</sup>). 
    - We set 100 trials for the bootstrap tests.
    - `-intensity_bootstrap True` means we are using bootstrap to probe the uncertainty.
    - `-intensity_file`, the path of the intensity file (note: It should be dedispersed to an initial DM value and the RFI should be masked beforehand)
    - Finally, save the results in a npz file.

#### Comparing the result of DM and the uncertainty with [the DM-phase algorithm](https://github.com/danielemichilli/DM_phase)
- `mpirun -np 100 python  DM_phase_test.py -data_path '/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/sim3_individual_test.npz' -save_path '/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/DM_phase_sim3_individual.npz'`
- `-data_path` the dedispersed data at an initial DM with RFI masked. 
- `-save_path` saving the result to a npz file.


## The output of the algorithm
### During the analysis
* The number of DM steps, which the analysis is finished, will be reported.

### At the end of the analysis
* The optimized DM in pc cm<sup>-3</sup> will be shown. 
* The optimized DM uncertainty in pc cm<sup>-3</sup> will be shown. 

## Contact information
* Please report issues [here](https://github.com/hsiuhsil/DM_power/issues).
* Please feel free to contact Hsiu-Hsien Lin at `hsiuhsil@alumni.cmu.edu` for further discussions.
