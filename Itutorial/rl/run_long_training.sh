#!/usr/bin/env bash
# Long F1Tenth training on mini_oval_flat — run in ~30 min chunks,
# resuming from the previous checkpoint each time. Targets ~3 hours total.
#
# Writes checkpoints to Itutorial/rl/ckpt_f1tenth_chunk{N}.pt and
# prints a "CHUNK_OK" marker so a wrapper loop can detect completion.
#
# Usage: bash Itutorial/rl/run_long_training.sh <chunk_id> [--resume <prev_ckpt>]

set -e
CHUNK=${1:-1}
RESUME_FLAG=""
if [ "$2" = "--resume" ] && [ -n "$3" ]; then
    RESUME_FLAG="--resume $3"
fi

CKPT="Itutorial/rl/ckpt_f1tenth_chunk${CHUNK}.pt"

# Multi-env at 256 (GPU has ~15 GB free)
OMNI_KIT_ACCEPT_EULA=YES /home/js/anaconda3/envs/hmclab_test/bin/python \
    Itutorial/rl/train.py \
    --robot f1tenth \
    --num_envs 256 \
    --rollout_steps 48 \
    --iters 1800 \
    --lr 2e-4 \
    --ckpt "$CKPT" \
    $RESUME_FLAG

echo "CHUNK_OK chunk=${CHUNK} ckpt=$CKPT"
