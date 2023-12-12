#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=gen_emb ## CHANGE JOBNAME HERE
#SBATCH --array=0

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH --ntasks-per-node 1                         ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=0-1:00:00
#SBATCH --gres=gpu:rtx_2080:1
##SBATCH --exclude=node004,node005,node006,node008,node901,node902,node912,node913,node914
# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=hyen@princeton.edu

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Array Job ID                   = $SLURM_ARRAY_JOB_ID"
echo "Array Task ID                  = $SLURM_ARRAY_TASK_ID"
echo "Cache                          = $TRANSFORMERS_CACHE"

PORT=$(shuf -i 30000-65000 -n 1)
echo "Port                           = $PORT"

export OMP_NUM_THREADS=8

IDX=$SLURM_ARRAY_TASK_ID
NGPU=$SLURM_GPUS_ON_NODE

if [[ -z $IDX ]]; then IDX=0; fi
if [[ -z $NGPU ]]; then NGPU=1; fi

conda activate ca

export TAG=final
echo "Tag                            = $TAG"

ARCH=ProtGPT2
CONFIGS=(${ARCH}_51m ${ARCH}_65m ${ARCH}_82m ${ARCH}_97m ${ARCH}_112m ${ARCH}_124m ${ARCH}_146m ${ARCH}_167m)

CONFIG=${CONFIGS[$IDX % 8]}

echo "Config                         = $CONFIG"

LRs=(5e-4 1e-3 5e-3)
LR=${LRs[$IDX / 8]}

TOTAL_BS=2048
GRAD_ACC=128
SEED=42

OUTPUT_DIR=output/$CONFIG-$TAG-lr$LR-bs$TOTAL_BS-gc$GRAD_ACC-$SEED

echo "Output directory               = $OUTPUT_DIR"

for DATA in "protein_sequence_records_df.csv SKEMPI_seq.csv"; do
    python generate_embeddings.py \
        --model_path $OUTPUT_DIR \
        --data_path $DATA \
        --output_path $OUTPUT_DIR/$DATA
done

wait;

