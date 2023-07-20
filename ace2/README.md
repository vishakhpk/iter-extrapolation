## ACE2

1. The original dataset of ACE2 proteins is [here](https://console.cloud.google.com/storage/browser/sfr-amadani-conference-data/genhance), obtained from the [Genhance repository](https://github.com/salesforce/genhance). We use these to train the scorer. This script consumes the pickle file from the ACE2 data and creates the train and validation json files to train the scorer. 
```
python3 create_scorer_dataset.py
```
2. The training script for the scorer is largely identical to the [example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) training script from HuggingFace with minor changes to handle the ACE2 train data. We finetune the [prot\_bert](https://huggingface.co/Rostlab/prot_bert) model to function as the scorer.
```
sh run_train_scorer.sh
```
3. For creating the paired data, we randomly mask out spans of the wildtype sequence of the task and infill these with a [prot\_T5\_XL](https://huggingface.co/Rostlab/prot_t5_xxl_uniref50) model to create minimal pairs. The output of this script is a JSON file with the masked out sequences and generated mutants from MLM.
```
python3 mlm_mutation.py
```
4. The next step is to score the output mutants generated in Step 3 using the trained scorer from Step 2 in order to create the training data for the generator. TODO: Discriminator -> Generator data using the scorer
5. Finally we train the generator on the created paired data. This again is largely a small variation in the Huggingface [example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py) script with the changes mainly around data loading. We finetune [prot\_T5\_XL](https://huggingface.co/Rostlab/prot_t5_xxl_uniref50) for the task.  
```
sh run_train_parallel.sh
```
