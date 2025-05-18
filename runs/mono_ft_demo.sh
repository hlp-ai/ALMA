OUTPUT_DIR=${1:-"./Qwen2.5-0.5B"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"

accelerate launch --config_file configs/deepspeed_train_config_demo.yaml \
     run_llmmt.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --oscar_data_path nthngdy/oscar-small \
    --oscar_data_lang zh,is \
    --interleave_probs "0.7,0.3" \
    --streaming \
    --max_steps 600000 \
    --do_train \
    --low_cpu_mem_usage \
    --fp16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 1 \
    --output_dir ${OUTPUT_DIR} \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --seed 42 \
    --overwrite_output_dir \
    --report_to none
