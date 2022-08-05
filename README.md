# DM_power
## The purpose of this algorithm 
The algorithm to optimize dispersion measure (DM) for Fast Radio Bursts (FRBs) by maximizing the sub-structure in the power spectrum 
* The link of the publication (TBA)

## The usage of the algorithm

### The required intensity file

* A `npy` file with a shape of (frequency, time)
* The data should be dedispersed to an initial DM value. The algorithm will determine the optimized DM value by given parameters.
* The radio-frequency-interference (RFI) should be removed already.

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
- `python DM_power.py -bw 200 -f0 550 -nchan 2048  -dt 0.00032768 -dm_start -2 -dm_end 2 -dm_steps 51 -trials 1000 -intensity_file 'example.npy'`
    - We use the data with 2048 frequency channels from 550 to 750 MHz with a timing resolution of 0.00032768 seconds.
    - The range of the optimized DM value, different from the initially dedispersed data, is from -2 to 2 (pc cm<sup>-3</sup>). 
    - We set 1000 trials for the bootstrap tests.
    - The `example.npy` is an intensity file, which is dedispersed to an inital DM value and the RFI is removed.
 
## The output of the algorithm
### During the analysis
* The number of DM steps, which the analysis is finished, will be reported.

### At the end of the analysis
* The optimized DM in pc cm<sup>-3</sup> will be shown. 
* The optimized DM uncertainty in pc cm<sup>-3</sup> will be shown. 
