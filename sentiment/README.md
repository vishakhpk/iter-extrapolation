## Sentiment Analysis

The scorer for sentiment analysis is a trained RoBERTa model that predicts Yelp review scores from within the training region. train\_scorer.py filters the Yelp data to only contain examples from the training region.
```
sh run_train_scorer.sh
```
We then mask out spans of text in the examples and fill them in with masked language modeling in order to obtain minimal pairs. Here the last argument is an index to save the files so that data creation can be parallelized.
```
python3 denoise_create_mask.py yelp data_directory/ 1 
```
This is followed by scoring the created pairs using the trained scorer.
```
python3 denoise_and_score.py yelp data_directory/ ./yelp_trained_scorer/ 1
```
```
python3 create_json_file.py data_directory/denoise_final.pkl data_directory 
```
Once the paired data is scored to obtain the control codes, we can train the generator model on this:
```
sh run_train_generator.sh
```
