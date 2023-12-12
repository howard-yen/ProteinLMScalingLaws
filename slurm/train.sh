#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=plm ## CHANGE JOBNAME HERE
#SBATCH --array=20-23

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH --ntasks-per-node 1                         ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2-0:00:00
#SBATCH --gres=gpu:a6000:2
#SBATCH --exclude=node004,node005,node006,node008,node901,node902,node912,node913,node914
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

export TAG=v2
echo "Tag                            = $TAG"

ARCH=ProtGPT2
STEPS=(10000 7026 5123 3941 3202 2966 2373 1977) # gpt2

#ARCH=ProtLlama2
#STEPS=(10000 7728 6069 5052 4327 3905 3291 2844) # llama

CONFIGS=(${ARCH}_51m ${ARCH}_65m ${ARCH}_82m ${ARCH}_97m ${ARCH}_112m ${ARCH}_124m ${ARCH}_146m ${ARCH}_167m)

CONFIG=${CONFIGS[$IDX % 8]}
STEPS=${STEPS[$IDX % 8]}
SSTEP=$(expr $STEPS / 5)

echo "Config                         = $CONFIG"

LRs=(5e-4 1e-3 5e-3)
LR=${LRs[$IDX / 8]}

TOTAL_BS=2048
# 8 is ok for seq length 1024 on a6000, but 16 is too much
TRAIN_BS=8
GRAD_ACC=$(expr $TOTAL_BS / $NGPU / $TRAIN_BS)
WARMUP=0.04
SEED=42

OUTPUT_DIR=output/$CONFIG-$TAG-lr$LR-bs$TOTAL_BS-gc$GRAD_ACC-$SEED

echo "Output directory               = $OUTPUT_DIR"
echo "Eval/Save steps                = $SSTEP"

# comment out --model_name_or_path to random initialize
torchrun --nproc_per_node $NGPU --master_port $PORT run_clm.py \
    --config_name configs/$CONFIG.json \
    --tokenizer_name nferruz/ProtGPT2 \
    --dataset_name nferruz/UR50_2021_04 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $TRAIN_BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate $LR \
    --save_steps $SSTEP \
    --save_total_limit 5 \
    --optim "adamw_torch" \
    --torch_dtype bfloat16 \
    --bf16 True \
    --lr_scheduler_type "cosine" \
    --evaluation_strategy "steps" \
    --eval_steps $SSTEP \
    --logging_steps 10 \
    --warmup_ratio $WARMUP \
    --max_steps $STEPS \
    --dataloader_num_workers 8 \
    --ddp_find_unused_parameters False \
    --max_eval_samples 1000 \
    --cache_dir cache \
    --output_dir $OUTPUT_DIR

wait;

