#!/bin/bash
set -e

echo "=== FFW SG2 파인튜닝 시작 ==="

export PYTHONPATH=/workspace/gr00t:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DeepSpeed 단일 GPU 환경 변수 설정
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

cd /workspace/gr00t

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /workspace/data/ffw_sg2_rev1_task_330_0205_jungmin_1 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /workspace/gr00t/examples/FFW_SG2/ffw_sg2_config.py \
    --num-gpus 1 \
    --output-dir /workspace/outputs/ffw_sg2_finetune \
    --global-batch-size 4 \
    --gradient-accumulation-steps 32 \
    --learning-rate 1e-4 \
    --max-steps 5000 \
    --save-steps 500 \
    --save-total-limit 5 \
    --dataloader-num-workers 2 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --weight-decay 1e-5 \
    --warmup-ratio 0.05
