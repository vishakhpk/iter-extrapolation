## Code and models for the paper "Extrapolative Controlled Sequence Generation via Iterative Refinement"
This repo contains the accompanying code for [Iterative Controlled Extrapolation](https://arxiv.org/abs/2303.04562). In this work, we study the problem of extrapolative controlled generation, i.e., generating sequences with attribute values beyond the range seen in training. Our method aims to iteratively make local edits to sequences to enable extrapolation. 

This repo is organized into three directories for each task, [sentiment control](https://github.com/vishakhpk/iter-extrapolation/tree/main/sentiment), [ACE2 stability](https://github.com/vishakhpk/iter-extrapolation/tree/main/ace2) and [AAV fitness](https://github.com/vishakhpk/iter-extrapolation/tree/main/aav). Each directory has an individual readme and code to create the paired training data for the generator as well as training code for the scorer and generator.

## Steps
1. **Scorer Training:** Train the scorer using data from the training region
2. **Paired Data Creation**: Run MLM on the input sequences to obtain a large number of minimal pairs. Then score these pairs using the trained scorer to obtain the required control tag. 
3. **Generator Training**: Train the generator by finetuning a T5 model on the generated paired data
4. **Inference**: Run scorer-guided or scorer-free inference to obtain the output sequences

## Citation
```bibtex
@inproceedings{padmakumar2023extrapolative,
    author={Vishakh Padmakumar and Richard Yuanzhe Pang and He He and Ankur P Parikh},
    title={Extrapolative Controlled Sequence Generation via Iterative Refinement },
    booktitle={International Conference on Machine Learning (ICML)},
    year={2023}
}
```

## TODOs
1. Update model weights for each task
2. Upload datasets used for training models 
