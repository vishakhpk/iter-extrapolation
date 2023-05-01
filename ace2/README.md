## ACE2

The original dataset of ACE2 proteins is [here](https://console.cloud.google.com/storage/browser/sfr-amadani-conference-data/genhance), obtained from the [Genhance repository](https://github.com/salesforce/genhance). We use these to train the scorer.
```
python3 create_scorer_dataset.py
```
```
sh run_train_scorer.sh
```
For creating the paired data, we used masked language modeling with a prot\_T5\_XXL model 
```
python3 mlm_mutation.py
```
TODO: Discriminator -> Generator data using the scorer
```
sh run_train_parallel.sh
```
