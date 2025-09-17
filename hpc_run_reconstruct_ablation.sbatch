#!/bin/bash
# 
#SBATCH -J movie_reconstruction # job name
#SBATCH -p a100 # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 32G # memory pool for all cores (was 32)
#SBATCH -n 4 # number of cores (was 4)
#SBATCH -t 30-00:00 # time (D-HH:MM)
#SBATCH --gres gpu:a100:1 # request 1 GPU a100 or rtx5000
#
#SBATCH --array=0-6%4
#
#SBATCH -o /ceph/margrie/joelb/video_reconstruction_from_sensorium2023_winner/hpc_runs/hpc_outputs/.%x.%N.%j.%A.%a.out # STDOUT
#SBATCH -e /ceph/margrie/joelb/video_reconstruction_from_sensorium2023_winner/hpc_runs/hpc_errors/.%x.%N.%j.%A.%a.err # STDERR


# print the job id
echo "Starting job $SLURM_ARRAY_TASK_ID"

# define paths
WORKING_DIR=/root/video_reconstruction_from_sensorium2023_winner
SINGULARITY_IMAGE=/ceph/margrie/joelb/video_reconstruction_from_sensorium2023_winner/sensorium.sif
SINGULARITY_BINDPATH=/ceph/margrie/joelb/video_reconstruction_from_sensorium2023_winner:$WORKING_DIR

# load cuda
module load cuda/12.0

# run the script
cd /ceph/margrie/joelb/video_reconstruction_from_sensorium2023_winner
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --population_reduction 0.5
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --population_reduction 0.75
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --population_reduction 0.875

# add replacement trials
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 0 --start_trial 10 --end_trial 12 --population_reduction 0.5
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 1 --start_trial 10 --end_trial 12 --population_reduction 0.5
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 2 --start_trial 10 --end_trial 11 --population_reduction 0.5
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 2 --start_trial 14 --end_trial 17 --population_reduction 0.5
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 3 --start_trial 10 --end_trial 11 --population_reduction 0.5
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 0 --start_trial 10 --end_trial 12 --population_reduction 0.75
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 1 --start_trial 10 --end_trial 12 --population_reduction 0.75
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 2 --start_trial 10 --end_trial 11 --population_reduction 0.75
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 2 --start_trial 14 --end_trial 17 --population_reduction 0.75
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 3 --start_trial 10 --end_trial 11 --population_reduction 0.75
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 0 --start_trial 10 --end_trial 12 --population_reduction 0.875
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 1 --start_trial 10 --end_trial 12 --population_reduction 0.875
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 2 --start_trial 10 --end_trial 11 --population_reduction 0.875
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 2 --start_trial 14 --end_trial 17 --population_reduction 0.875
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list $SLURM_ARRAY_TASK_ID --animals 3 --start_trial 10 --end_trial 11 --population_reduction 0.875



