# Movie_reconstruction
Code to produce figures from [Bauer et al., (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.06.19.599691v3).
* To reproduce the figures from the paper, download this code and the reconstructions files. Then create an environment using the requirements_analysis.txt file and run the jupyter notebooks (see below for details).
    * [reconstructions](https://gin.g-node.org/Joel-Bauer/Movie_reconstruction)
* To run reconstructions from scratch, download the code for the winning model of the Sensorium competition (DwiseNeuro) and add the code from this repository into the same folder. Optionally, the re-trained weights and alpha masks used in the paper can be downloaded. To re-train the model from scratch you will need both the old and new Sensorium dataset, but if you use the retrained weights, you only need the new dataset. To reconstruct the Gaussian noise or drifting grating stimuli those will also need to be downloaded. Then convert the docker container from the DwiseNeuro repository to a singularity container and run the .sbatch scripts (see below for details).
    * [DwiseNeuro](https://github.com/lRomul/sensorium)
    * Sensorium 2023 data [old](https://gin.g-node.org/pollytur/Sensorium2023Data) and [new](https://gin.g-node.org/pollytur/sensorium_2023_dataset)
    * Drifting grating and Gaussian noise [stimuli](https://gin.g-node.org/Joel-Bauer/Movie_reconstruction)
    * [Retrained weights and alpha masks from the paper]( https://gin.g-node.org/Joel-Bauer/Movie_reconstruction)


# Run analysis with downloaded reconstructions
To run the jupyter notebooks below first create an environment using requiremnts_analysis.txt

## To analyse results of the natural movie reconstruction (and ensembling effect)
run analyse_reconstructions.ipynb
This will produce figure panels for Fig 1-3 and supplementary Fig 1 & 3. 

## To analyse results of the population ablation experiment
run analyse_reconstructions_population_reduction.ipynb
This will produce figure panels from Fig 5.

## To analyse results of the Gaussian noise stimuli reconstruction
run analyse_reconstructions_gaussian_noise.ipynb
This will produce figure panels from Fig 4.

## To analyse results of the drifting grating stimuli reconstruction
run analyse_reconstructions_drifting_gratings.ipynb
This will produce figure panels from supplementary Fig 2.

# Run reconstructions from scratch
The .sbatch files described below are designed to run the code on an HPC. To execute them as they are convert the docker container file to a singularity container. 
```
docker save -o lRomul_sensorium.tar sensorium:latest
```

## To re-train DwiseNeuro (SOTA dynamic neural encoding model by Ruslan Baikulov)
Run hpc_run_train.sbatch to train 7 instances of the DwiseNeuro model using the same data splits. This will ensure that none of the model instances are trained on fold0. fold0 is then used for movie reconstruction. 
The weights can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/data/experiments/true_batch_002/

## Train the transparency masks
To generate alpha masks for each mouse from scratch use the code below:
```
cd <mydirectory>
singularity shell --nv --bind <mydirectory>:/root/sensorium sensorium.sif
python3 scripts/reconstruct_videos_natural.py
```
These masks represent the population receptive field, i.e. the area of the video frame which influences the activity of neurons in the population.
The alpha masks can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/masks/

## Reconstruct natural movies from recorded neural activity
Run the batch script to reconstruct 10 movies each from 5 mice separately using 7 instances of the DwiseNeuro model. 
```
sbatch hpc_run_reconstructions.sbatch
```
The reconstructions can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/

## Reconstruct natural movies with reduced population sizes (population ablation)
To test the effect of neural population size on movie reconstruction quality we performed an ablation experiment.
```
sbatch hpc_run_reconstructions_ablation.sbatch
```
The reconstructions can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/

## To reconstruct Gaussian noise stimuli from predicted neural activity
To test the resolution limits of the reconstruction method we reconstructed Gaussian noise stimuli at a range of spatial and temporal length constants from the predicted activity of the DwiseNeuro model. 
Download the stimuli from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/utils_reconstruction/gaussian_noise_movies.npz
```
sbatch hpc_run_reconstructions_noise.sbatch
```
The reconstructions can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/

## To reconstruct drifting grating stimuli from predicted neural activity
to test the resolution limits of the method we also reconstructed drifting gratings
Download the stimuli from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/utils_reconstruction/grating_movies.npz
```
sbatch hpc_run_reconstructions_gratings.sbatch
```
The reconstructions can also be downloaded from https://gin.g-node.org/Joel-Bauer/Movie_reconstruction/reconstructions/

