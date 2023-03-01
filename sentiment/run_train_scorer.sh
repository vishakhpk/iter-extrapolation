# export TASK_NAME=yelp_review_full
export TASK_NAME=yelp_scorer

python train_scorer.py \
  --model_name_or_path roberta-large \
  --resume_from_checkpoint roberta-large \
  --dataset_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --save_strategy steps \
  --save_steps 1000 \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-6 \
  --weight_decay 0.00001 \
  --num_train_epochs 15 \
  --output_dir ./trained_yelp_scorer/ \
  --cache_dir ./cache/ \
  --load_best_model_at_end
