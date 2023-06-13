#!/bin/bash 
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=25
#SBATCH --time=3:00:00
#SBATCH --job-name=mpi_job
#SBATCH --output=mpi_output_%j.txt
#SBATCH --mail-type=FAIL

cd /scratch/p/pen/hsiuhsil/DM_power

# 100 bootstrap on the noiseamp R3 and B0329 files
## b11 of R3
mpirun -np 100 python DM_power.py -bw 200 -f0 550 -nchan 2048 -dt 0.00032768 -dm_start -2 -dm_end 2 -dm_steps 51 -trials 100 -rescaled False  -intensity_bootstrap True -intensity_file "/scratch/p/pen/hsiuhsil/DM_power/noiseamp_files/burst_11_noiseamp_0.npy" -save_path "/scratch/p/pen/hsiuhsil/DM_power/noiseamp_files/burst_11_noiseamp_0_bootstrap_test.npz"
## b0329 pulse
mpirun -np 100 python DM_power.py -bw 200 -f0 550 -nchan 2048 -dt 0.00008192 -dm_start -2 -dm_end 2 -dm_steps 51 -trials 100 -rescaled False  -intensity_bootstrap True -intensity_file "/scratch/p/pen/hsiuhsil/DM_power/noiseamp_files/burst_100_noiseamp_0.npy" -save_path "/scratch/p/pen/hsiuhsil/DM_power/noiseamp_files/burst_100_noiseamp_0_bootstrap_test.npz"

# Generate the simulated Gaussian functions.
mpirun -np 100 python DM_power.py -bw 200 -f0 550 -nchan 2048 -dt 0.00032768 -dm_start -2 -dm_end 2 -dm_steps 51 -trials 100 -rescaled False  -intensity_sim True -sim_width 64 -sim_mu 0 -sim_sigma 0.25 -save_path "/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/sim1_individual_test.npz"

# getting DM and the unvertainty of the simulated files.
mpirun -np 100 python DM_power.py -bw 200 -f0 550 -nchan 2048 -dt 0.00032768 -dm_start -2 -dm_end 2 -dm_steps 51 -trials 100 -rescaled False  -intensity_bootstrap True -intensity_file "/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/sim1_single_profile.npy" -save_path "/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/sim1_bootstrap_test.npz"

# test 100 individual profiles for DM-phase
mpirun -np 100 python  DM_phase_test.py -data_path '/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/sim3_individual_test.npz' -save_path '/scratch/p/pen/hsiuhsil/DM_power/sim_gau_pulses/DM_phase_sim3_individual.npz'
