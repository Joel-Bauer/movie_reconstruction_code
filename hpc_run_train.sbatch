#!/bin/bash
# 
#SBATCH -J sensorium_video_recon # job name
#SBATCH -p a100 # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -n 8 # number of cores
#SBATCH -t 2-0:00 # time (D-HH:MM)
#SBATCH --gres gpu:a100:1 # request 1 GPU a100 or rtx5000
#
#SBATCH --array=0-6%4
#
#SBATCH -o /ceph/margrie/joelb/video_reconstruction_from_sensorium2023_winner/hpc_runs/hpc_outputs/.%x.%N.%j.out # STDOUT
#SBATCH -e /ceph/margrie/joelb/video_reconstruction_from_sensorium2023_winner/hpc_runs/hpc_errors/.%x.%N.%j.err # STDERR


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
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/train_test_splitout.py -e true_batch_002 -f $SLURM_ARRAY_TASK_ID


