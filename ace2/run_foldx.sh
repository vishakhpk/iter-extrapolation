folder="./eval/"
ending="iter-output.tsv"
python foldx_stability_eval_new.py \
    --repair_pdb_dir ./repaired_single \
    --foldx_batch_size 500 \
    --foldx ../../foldx_5_2023/foldx_20231231 \
    --workers 16 \
    -i $folder/${ending} \
    -o $folder/sampled/

