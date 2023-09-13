## AAV

The scorer used for AAV is the CNN model trained on the low-vs-high split of the task as obtained from the [FLIP benchmark](https://github.com/J-SNACKKB/FLIP/tree/main/splits/aav). The script to train the CNN is found [here](https://github.com/J-SNACKKB/FLIP/tree/main/baselines) and can run as follows: 
```
python3 train_all.py --split aav_6 --model cnn --gpu 0 
```
TODO: Include MLM code

Score sequences with trained CNN model. This script has a lot of dependencies from the FLIP codebase and should be copied into the baselines/ directory of FLIP to execute it.
```
python3 score_sequences.py
```
Create pairs from the scored CNN data. 
```
python3 create_pairs.py scored-train.json generator_data/
```
Train the generator model
```
sh run_train_generator.sh
```

Data for training the generator follows the tranlsation JSON format for HuggingFace. We make our train set of 900k sequences available [here](https://drive.google.com/file/d/1FOXwjloxwHf7rkMn5n_WzP6E3TUjN1_k/view?usp=drive_link) as a tarball.  
