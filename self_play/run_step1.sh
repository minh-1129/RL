#!/bin/bash
set -e
MODEL_PATH=$1
OUTPUT_DIR=$2
RUN_NAME=$3
STEPS=$4

export CUDA_VISIBLE_DEVICES=0

accelerate launch \
    --config_file ./deepspeed/no_zero.yaml --num_processes 1 \
    step1_generator.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 2 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --run_name "$RUN_NAME" \
    --number_of_generated_data 2000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --use_peft \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_completion_length 2048 \
    --log_completions \
    --num_generations 4 \
    --remove_unused_columns False \
    --loss_type grpo \
    --report_to wandb

python merge_lora.py \
    "$MODEL_PATH" \
    "$OUTPUT_DIR/adapter-final" \
    "$OUTPUT_DIR/merged-$STEPS"
