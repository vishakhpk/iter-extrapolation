BSZ=48

python3 train_model_parallel.py \
    --model_name_or_path prot_t5_xl_uniref50 \
    --do_train \
    --do_eval \
    --source_lang src \
    --target_lang tgt \
    --train_file ./data_directory/train.json \
    --validation_file ./data_directory/val_small.json \
    --output_dir ./generator_model/ \
    --per_device_train_batch_size=$BSZ \
    --per_device_eval_batch_size=$BSZ \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 4500  \
    --save_steps 4500 \
    --max_source_length 128 \
    --max_target_length 128 \
    --val_max_target_length 128 \
    --cache_dir ./cache-large 

# \

    # --load_best_model_at_end 
    # --model_name_or_path prot_t5_xl_uniref50 \
    # --resume_from_checkpoint ./run_parallel_16/checkpoint-20000 \
