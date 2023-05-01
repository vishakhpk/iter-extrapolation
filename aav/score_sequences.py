import numpy as np
import copy
import random
import sys
import json
from pathlib import Path
from filepaths import * 

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import re 
from csv import writer

sys.path.append(BASELINE_DIR)
import numpy as np
from utils import *
from evals import *
from models import * 
from train import * 

import argparse 

split_dict = {
    'aav_1': 'des_mut' ,
    'aav_2': 'mut_des',
    'aav_3': 'one_vs_many',
    'aav_4': 'two_vs_many',
    'aav_5': 'seven_vs_many',
    'aav_6': 'low_vs_high',
    'aav_7': 'sampled',
    'meltome_1' : 'mixed_split',
    'meltome_2' : 'human',
    'meltome_3' : 'human_cell',
    'gb1_1': 'one_vs_rest',
    'gb1_2': 'two_vs_rest',
    'gb1_3': 'three_vs_rest',
    'gb1_4': 'sampled',
    'gb1_5': 'low_vs_high'
}

def create_parser():
    parser = argparse.ArgumentParser(description="train esm")
    parser.add_argument("--split", type=str)
    parser.add_argument("--model", choices = ["ridge", "cnn", "esm1b", "esm1v", "esm_rand"], type = str)
    parser.add_argument("--gpu", type=str, nargs='?', default='0')
    parser.add_argument("--output_index", type=str, default = '2')
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--mut_mean", action="store_true")
    parser.add_argument("--flip", action="store_true") # for flipping mut-des and des-mut
    parser.add_argument("--ensemble", action="store_true") 
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--gb1_shorten', action="store_true")

    return parser

def model_eval(model_path):
    
    ip = []
    with open("./train.json", "r") as f:
        for line in f:
            ip.append(json.loads(line.strip()))
    print(len(ip))
    texts = [item['text'].replace(" ", "") for item in ip]
    scores = [item['label'] for item in ip]

    # mut = "M A A D G Y L P D W L E D T L S E G I R Q W W K L K P G P P P P K P A E R H K D D S R G L V L P G Y K Y L G P F N G L D K G E P V N E A D A A A L E H D K A Y D R Q L D S G D N P Y L K Y N H A D A E F Q E R L K E D T S F G G N L G R A V F Q A K K R V L E P L G L V E E P V K T A P G K K R P V E H S P V E P D S S S G T G K A G Q Q P A R K R L N F G Q T G D A D S V P D P Q P L G Q P P A A P S G L G T N T M A T G S G A P M A D N N E G A D G V G N S S G N W H C D S T W M G D R V I T T S T R T W A L P T Y N N H L Y K Q I S S Q S G A S N D N H Y F G Y S T P W G Y F D F N R F H C H F S P R D W Q R L I N N N W G F R P K R L N F K L F N I Q V K E V T Q N D G T T T I A N N L T S T V Q V F T D S E Y Q L P Y V L G S A H Q G C L P P F P A D V F M V P Q Y G Y L T L N N G S Q A V G R S S F Y C L E Y F P S Q M L R T G N N F T F S Y T F E D V P F H S S Y A H S Q S L D R L M N P L I D Q Y L Y Y L S R T N T P S G T T T Q S R L Q F S Q A G A S D I R D Q S R N W L P G P C Y R Q Q R V S K T S A D N N N S E Y S W T G A T K Y H L N G R D S L V N P G P A M A S H K D D E E K F F P Q S G V L I F G K Q G S E K T N V D I E K V M I T D E E E I R T T N P V A T E Q Y G Q C V C S W E H Q G G S N Q Y Q A A T A D V N T Q G V L P G M V W Q D R D V Y L Q G P I W A K I P H T D G H F H P S P L M G G F G L K H P P P Q I L I K N T P V P A N P S T T F S A A K F A S F I T Q Y S T G Q V S V E I E W E L Q K E N S K R W N P E I Q Y T S N Y N K S V N V D F T V D T N G V Y S E P R P I G T R Y L T R N L"
    # mut = mut.replace(" ", "")
    collate = ASCollater(vocab, Tokenizer(vocab), pad=True)
    batch_size = 30 # smaller batch sizes for meltome since seqs are long
    test = pd.DataFrame.from_dict({'sequence':texts, 'target':scores, 'validation':[None for i in range(len(texts))], 'set':["test" for i in range(len(texts))]})
    test_set = SequenceDataset(test)
    # breakpoint()
    test_iterator = DataLoader(test_set, collate_fn=collate, batch_size=30)
    # breakpoint()
    # initialize model
    cnn_model = FluorescenceModel(len(vocab), args.kernel_size, args.input_size, args.dropout) 
    device = torch.device('cuda:'+args.gpu)
    cnn_model = cnn_model.to(device)
    # bestmodel_save = MODEL_PATH / 'bestmodel.tar' 
    sd = torch.load(model_path)
    cnn_model.load_state_dict(sd['model_state_dict'])
    print('loaded the saved model from ', model_path)
    # breakpoint()

    def test_step(model, batch):
        src, tgt, mask = batch
        src = src.to(device).float()
        tgt = tgt.to(device).float()
        mask = mask.to(device).float()
        # breakpoint()
        output = model(src, mask)
        return output.detach().cpu(), tgt.detach().cpu()
    
    cnn_model = cnn_model.eval()
    outputs = []
    tgts = []
    for i, batch in enumerate(test_iterator):
        # print(batch)
        output, tgt = test_step(cnn_model, batch)
        # print(output, tgt)
        outputs += [item[0] for item in output.tolist()]
        tgts += [item[0] for item in tgt.tolist()]
        # breakpoint()
        # break

    op = []
    for i in range(len(outputs)):
        # print(i)
        # print(ip[i])
        # print(outputs[i], tgts[i])
        op_item = copy.deepcopy(ip[i])
        op_item['label'] = outputs[i]
        op.append(op_item)
        # op_item['matching_label'] = scores[i]
        # assert round(op_item['label'], 2) == round(tgts[i], 2), print(i, op_item, tgts[i])

    with open("scored-train.json", "w") as f:
        for item in op:
            f.write(json.dumps(item)+'\n')


def main(args):
    model_eval("./path/to/saved/cnn/bestmodel.tar") 

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
