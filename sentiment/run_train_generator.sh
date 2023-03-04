BSZ=4

python3 train_model.py \
    --model_name_or_path t5-base \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --source_lang src \
    --target_lang tgt \
    --train_file ./data_directory/ \
    --validation_file ./data_directory/ \
    --output_dir ./generator_model/ \
    --per_device_train_batch_size=$BSZ \
    --per_device_eval_batch_size=$BSZ \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 5000  \
    --save_steps 5000 \
    --max_source_length 512 \
    --max_target_length 512 \
    --val_max_target_length 512 \
    --cache_dir ./cache-large 

# \
    # --load_best_model_at_end 
