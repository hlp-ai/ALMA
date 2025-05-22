OUTPUT_DIR=${1:-"./Qwen2.5-0.5B-cpt-parallel-ft-lora"}
pairs=${2:-"zh-en,en-zh"}
LORA_RANK=${3:-"16"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"

accelerate launch --config_file configs/deepspeed_train_config_demo.yaml \
     run_llmmt.py \
    --model_name_or_path /home/liuxf/hdisk/llm/ALMA/Qwen2.5-0.5B-cpt \
    --tokenizer_name Qwen/Qwen2.5-0.5B \
    --mmt_data_path  ./human_written_data/ \
    --use_peft \
    --lora_rank ${LORA_RANK} \
    --do_train \
    --do_eval \
    --do_predict \
    --language_pairs ${pairs} \
    --low_cpu_mem_usage \
    --fp16 \
    --learning_rate 2e-3 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.1 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --eval_strategy steps \
    --eval_steps 0.2 \
    --save_strategy steps \
    --save_steps 0.2 \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --max_steps 1000 \
    --predict_with_generate \
    --prediction_loss_only \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --seed 42 \
    --overwrite_output_dir \
    --num_beams 5 \
    --ddp_timeout 999999 \
    --report_to none \
    --overwrite_cache \
    --load_best_model_at_end
    
## Evaluation (BLEU, COMET)
bash ./evals/eval_generation.sh ${OUTPUT_DIR} ${pairs}