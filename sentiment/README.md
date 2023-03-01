```
sh run_train_scorer.sh
```
```
python3 denoise_create_mask.py yelp data_directory/ 1 
```
```
python3 denoise_and_score.py yelp data_directory/ ./yelp_trained_scorer/ 1
```
```
python3 create_json_file.py data_directory/denoise_final.pkl data_directory 
```
