#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=eval_plm ## CHANGE JOBNAME HERE
#SBATCH --array=22-23

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH --ntasks-per-node 1                         ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=0-1:00:00
#SBATCH --gres=gpu:2
# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=vhchu@princeton.edu

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

if [[ -z $IDX ]]; then IDX=6; fi
if [[ -z $NGPU ]]; then NGPU=1; fi

module load anaconda3/2023.9
conda activate hky
cd /scratch/gpfs/vhchu/cos597n

export TAG=v2
echo "Tag                            = $TAG"

ARCH=ProtLlama2
CONFIGS=(${ARCH}_51m ${ARCH}_65m ${ARCH}_82m ${ARCH}_97m ${ARCH}_112m ${ARCH}_124m ${ARCH}_146m ${ARCH}_167m)

CONFIG=${CONFIGS[$IDX % 8]}

echo "Config                         = $CONFIG"

LRs=(5e-4 1e-3 5e-3)
LR=${LRs[$IDX / 8]}

TOTAL_BS=2048
GRAD_ACC=64
SEED=42


OUTPUT_DIR=output/$CONFIG-$TAG-lr$LR-bs$TOTAL_BS-gc$GRAD_ACC-$SEED

echo "Evaluating output directory               = $OUTPUT_DIR"

torchrun --nproc_per_node $NGPU --master_port $PORT run_clm.py \
    --dataset_name nferruz/UR50_2021_04 \
    --dataset_name datasetyay.hf \
    --model_name_or_path $OUTPUT_DIR \
    --do_train False \
    --do_eval True \
    --torch_dtype bfloat16 \
    --bf16 True \
    --dataloader_num_workers 8 \
    --ddp_find_unused_parameters False \
    --cache_dir cache \
    --per_device_eval_batch_size 32 \
    --max_eval_samples 100000 \
    --prediction_loss_only True \
    --output_dir $OUTPUT_DIR

for OD in "$OUTPUT_DIR/checkpoint-"*; do
    echo "Evaluating output dir $OD"
    if ! [[ -f $OD/all_results.json ]]; then
        torchrun --nproc_per_node $NGPU --master_port $PORT run_clm.py \
            --config_name configs/$CONFIG.json \
            --tokenizer_name nferruz/ProtGPT2 \
            --dataset_name nferruz/UR50_2021_04 \
            --dataset_name datasetyay.hf \
            --model_name_or_path $OD \
            --do_train False \
            --do_eval True \
            --torch_dtype bfloat16 \
            --bf16 True \
            --dataloader_num_workers 8 \
            --per_device_eval_batch_size 32 \
            --ddp_find_unused_parameters False \
            --cache_dir cache \
            --max_eval_samples 100000 \
            --prediction_loss_only True \
            --output_dir $OD
    else
        echo "all_results.json already exists"
    fi
done

wait;

