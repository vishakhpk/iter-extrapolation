## AAV

1. Data from the AAV task is obtained from the [FLIP benchmark](https://github.com/J-SNACKKB/FLIP/tree/main/splits/aav). The scorer for sequences is a CNN trained on the low-vs-high split of the task. The script to train the CNN is found [here](https://github.com/J-SNACKKB/FLIP/tree/main/baselines) and can be run as follows: 
```
python3 train_all.py --split aav_6 --model cnn --gpu 0 
```
2. Create perturbed sequences using the same MLM [script](https://github.com/vishakhpk/iter-extrapolation/blob/main/ace2/mlm_mutation.py) as ACE2, using the same pre-trained model. Details for the mutable region are found in the [paper](https://arxiv.org/abs/2303.04562).  
3. Score the generated sequences with the trained CNN scorer model. This script has a lot of dependencies from the FLIP codebase and should be copied into the baselines/ directory of FLIP to execute it.
```
python3 score_sequences.py
```
4. Create pairs of sequences with minimal change in AAV fitness score in order to train the generator model from the scored CNN data. 
```
python3 create_pairs.py scored-train.json generator_data/
```
The data we use for training the generator (900k sequence pairs) following the tranlsation JSON format for HuggingFace is available [here](https://drive.google.com/file/d/1FOXwjloxwHf7rkMn5n_WzP6E3TUjN1_k/view?usp=drive_link) as a tarball.  
5. Train the generator model. This again is largely a small variation in the Huggingface example [script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py) with the changes mainly around data loading. We finetune [prot\_T5\_XXL](https://huggingface.co/Rostlab/prot_t5_xxl_uniref50) for the task. One operational detail is that we truncate the sequences (from the start) to a fixed position due to GPU memory constraints. The truncated part of the sequences is unchanged in any mutations (the mutable region is not touched during truncation). So we just have to handle this case during inference by prepending any output from our generator with the same truncated head string before the generations are scored.
```
sh run_train_generator.sh
```
Model weights for a trained generator can be found [here](https://huggingface.co/vishakhpk/ice-aav-checkpoint-1).
6. We evaluate the output of the sequences using a CNN trained on the randomly sampled split of AAV. The script to train the CNN is found [here](https://github.com/J-SNACKKB/FLIP/tree/main/baselines) and can be run as follows: 
```
python3 train_all.py --split aav_7 --model cnn --gpu 0 
```
7. Running inference from the generated model works using the standard HuggingFace generation utilities. In the scorer-free inference method, we use the output from one iteration as the input for the next along with the control code to increase the fitness value. The script eval\_rewriter\_distribution.py generates mutations in this way from the wild type, using a model checkpoint saved from training. The output is saved in a pickle file using the arguments for the script as \<output directory\>/\<output filename\>-\<output model name\>.pkl.
```
python3 eval_rewriter_distribution.py <model directory> <output model name> <output directory> <output filename>
```
8. To run evaluation on the generated sequences, we include a utility script eval\_seqs\_cnn.py that consumes the output pickle file from the previous step and uses the trained CNN from step 6 to output fitness values. Here we show how to prepend the head string (that was truncated during data creation in step 5) to the generations and run inference. The scored output is saved as a pickle file. 
```
python3 eval_seqs_cnn.py <saved pickle file from step 7> <output file name>
``` 
