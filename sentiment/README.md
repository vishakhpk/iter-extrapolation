## Sentiment Analysis

1. The scorer for sentiment analysis is a trained RoBERTa model that predicts Yelp review scores from within the training region. train\_scorer.py filters the Yelp data to only contain examples from the training region. This is largely identical to the [example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) training script from HuggingFace with minor changes to handle the Yelp training data.
```
sh run_train_scorer.sh
```
2. We then mask out spans of text in the examples and fill them in with masked language modeling in order to obtain minimal pairs. Here the second argument is 'yelp' or 'sst5', we use the former for the experiments in the paper, the second argument is the output directory to be used to store output and the last argument is an index to save the files so that data creation can be parallelized. The output format is a pickled dictionary containing a pair of input and output lists where the output is a masked version of the input.  
```
python3 denoise_create_mask.py yelp data_directory/ 1 
```
3. The saved output from Step 2 is then consumed by the following script which denoises the masked sentences to create minimal pairs of sentences. We then score the created pairs using the trained scorer. Here the second argument refers to 'yelp' or 'sst' where we use the former for the paper, the second is the directory where the output of Step 2 was saved, the third argument is the directory of the trained scorer from Step 1 and the last argument is the same index from Step 2 so that the denoising and scoring can be parallelised. The output is again a saved pickled dictionary containing the input sequence, masked sequence, the created minimal pair and scores of each as assigned by the trained scorer.  
```
python3 denoise_and_score.py yelp data_directory/ ./yelp_trained_scorer/ 1
```
4. The output of Step 3 is consumed by this script to create json files which are used to train the generator model. We also assign the control tokens here to decide if the edit increases or decreases the associated sentiment of a sentence. 
```
python3 create_json_file.py data_directory/denoise_final.pkl data_directory 
```
5. Once the paired data is assigned the control codes, we can train the generator model on this. This again is largely a small variation in the Huggingface [example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py) script with the changes mainly around data loading. 
```
sh run_train_generator.sh
```
