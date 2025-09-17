#!/bin/bash
# 
#SBATCH -J movie_reconstruction # job name
#SBATCH -p a100 # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 32G # memory pool for all cores (was 32)
#SBATCH -n 4 # number of cores (was 4)
#SBATCH -t 4-00:00 # time (D-HH:MM)
#SBATCH --gres gpu:a100:1 # request 1 GPU rtx5000, rtx2080, a4500 or (a100 for partition a100)
#SBATCH --exclude="gpu-sr670-22, gpu-sr670-20"
#
#SBATCH --array=0-4%4
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
singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural_sparse.py --model_list 0 1 2 3 4 5 6 --animals $SLURM_ARRAY_TASK_ID

# add replacement trials
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list 0 1 2 3 4 5 6 --animals 0 --start_trial 10 --end_trial 12
    singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list 0 1 2 3 4 5 6 --animals 1 --start_trial 10 --end_trial 12
    singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list 0 1 2 3 4 5 6 --animals 2 --start_trial 10 --end_trial 11
    singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list 0 1 2 3 4 5 6 --animals 2 --start_trial 14 --end_trial 17
    singularity exec --nv --bind $SINGULARITY_BINDPATH --pwd $WORKING_DIR $SINGULARITY_IMAGE python3 scripts/reconstruct_videos_natural.py --model_list 0 1 2 3 4 5 6 --animals 3 --start_trial 10 --end_trial 11
fi