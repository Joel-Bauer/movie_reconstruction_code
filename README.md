# Movie_reconstruction
Code to produce figures from Bauer et al., 2024 (https://www.biorxiv.org/content/10.1101/2024.06.19.599691v3).
This codes functions as an extention to the repository https://github.com/lRomul/sensorium, merge both repositories to run this code.
To download the reconstructed videos, model weights, transparency masks, drifting grating stimuli, Gaussian noise stimuli go to https://gin.g-node.org/Joel-Bauer/Movie_reconstruction. 
This repository contains .sbatch scripts that runs the code in a singularity container, to convert the docker containiner from https://github.com/lRomul/sensorium run the code bellow.

'''
docker save -o lRomul_sensorium.tar sensorium:latest
'''

# Run reconstructions

## To re-train DwiseNeuro (SOTA dynamic neural encoding model by Ruslan Baikulov)
run hpc_run_train.sbatch
The weights can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/data/experiments/true_batch_002/

## To produce the transparency masks
For replicating the figures in the paper download the masks from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/masks/
To generate masks from scratch use the code below:

'''
cd <mydirectory>
singularity shell --nv --bind <mydirectory>:/root/sensorium sensorium.sif
python3 scripts/reconstruct_videos_natural.py
'''

## To reconstruct natural movies from recorded neural activity
run hpc_run_reconstructions.sbatch
The reconstructions can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/

## To reconstruct natural movies with reduced population sizes
run hpc_run_reconstructions_ablation.sbatch
The reconstructions can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/

## To reconstruct Gaussian noise stimuli from predicted neural activity
Download the stimuli from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/utils_reconstruction/gaussian_noise_movies.npz
run hpc_run_reconstructions_noise.sbatch
The reconstructions can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/

## To reconstruct drifting grating stimuli from predicted neural activity
Download the stimuli from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/utils_reconstruction/grating_movies.npz
run hpc_run_reconstructions_gratings.sbatch
The reconstructions can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/


# Run analysis
To run the jupyter notebooks below create an environment using requiremnts_analysis.txt

## To analyse results of natural movie reconstruction (and ensembling effect)
run analyse_reconstructions.ipynb

## To analyse results of population ablation experiment
run analyse_reconstructions_population_reduction.ipynb

## To analyse results of Gaussian noise stimuli reconstruction
run analyse_reconstructions_gaussian_noise.ipynb

## To analyse results of drifting grating stimuli reconstruction
run analyse_reconstructions_drifting_gratings.ipynb