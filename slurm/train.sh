#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=plm ## CHANGE JOBNAME HERE
#SBATCH --array=0-9

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH --ntasks-per-node 1                         ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:rtx_2080:4
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
if [[ -z $NGPU ]]; then IDX=1; fi

conda activate ca

export TAG=initial
echo "Tag                            = $TAG"

CONFIGS=(ProtGPT2_51m ProtGPT2_65m ProtGPT2_82m ProtGPT2_97m ProtGPT2_112m ProtGPT2_124m ProtGPT2_146m ProtGPT2_167m)
CONFIG=${CONFIGS[$IDX]}
echo "Config                         = $CONFIG"

LR=1e-4
TOTAL_BS=2048
# 8 is ok for seq length 1024 on a6000, but 16 is too much
TRAIN_BS=2
GRAD_ACC=$(expr $TOTAL_BS / $NGPU / $TRAIN_BS)
#STEPS=125000
STEPS=10000
WARMUP=0.04

OUTPUT_DIR=output/$CONFIG-$TAG

handle_signal()
{
    echo "$(date) Signal receive..."
    kill -s SIGUSR1 $PID
}
trap handle_signal SIGUSR1

# comment out --model_name_or_path to random initialize
torchrun --nproc_per_node $NGPU --master_port $PORT run_clm.py \
    --config_name configs/$CONFIG.json \
    --tokenizer_name gpt2 \
    --dataset_name nferruz/UR50_2021_04 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $TRAIN_BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate $LR \
    --save_steps 2500 \
    --save_total_limit 2 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --optim "adamw_torch" \
    --lr_scheduler_type "cosine" \
    --evaluation_strategy "steps" \
    --eval_steps 2500 \
    --logging_steps 10 \
    --warmup_ratio $WARMUP \
    --max_seq_length 1024 \
    --max_steps $STEPS \
    --dataloader_num_workers 8 \
    --ddp_find_unused_parameters False \
    --output_dir $OUTPUT_DIR

wait;

