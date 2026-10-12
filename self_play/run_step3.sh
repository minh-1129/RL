#!/bin/bash
set -e
MODEL_PATH=$1
DATA_PATH=$2
OUTPUT_DIR=$3
RUN_NAME=$4
STEPS=$5

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    step3_solver.py \
    --model_name_or_path "$MODEL_PATH" \
    --generated_data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 2 \
    --save_total_limit 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --beta 1e-2 \
    --bf16 True \
    --dtype bfloat16 \
    --use_peft \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_completion_length 4096 \
    --num_generations 4 \
    --remove_unused_columns False \
    --loss_type grpo \
    --logging_steps 1 \
    --run_name "$RUN_NAME" \
    --report_to wandb

python merge_lora.py \
    "$MODEL_PATH" \
    "$OUTPUT_DIR/adapter-final" \
    "$OUTPUT_DIR/merged-$STEPS"
