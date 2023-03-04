BSZ=8

python3 train_model_parallel.py \
    --model_name_or_path prot_t5_xl_uniref50 \
    --do_train \
    --source_lang src \
    --target_lang tgt \
    --train_file ./data_directory/train_cnn.json \
    --validation_file ./data_directory/val_cnn.json \
    --output_dir ./generator_model/ \
    --per_device_train_batch_size=$BSZ \
    --per_device_eval_batch_size=$BSZ \
    --save_steps 1000 \
    --max_source_length 512 \
    --max_target_length 512 \
    --val_max_target_length 512 \
    --overwrite_cache \
    --cache_dir ./cache-large 

