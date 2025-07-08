python llada/dynamic_length_sft.py \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --data_path llada/example_training_data.jsonl \
    --output_dir ./output/llada_dynamic \
    --batch_size 2 \
    --learning_rate 5e-6 \
    --num_epochs 2 \
    --max_steps 40 \
    --warmup_steps 5 \
    --gradient_accumulation_steps 8 \
    --save_steps 500 \
    --logging_steps 100 \
    --max_length 2048 \
    
    