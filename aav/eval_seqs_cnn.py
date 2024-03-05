import sys
from torch.utils.data import TensorDataset, DataLoader
from transformers import AlbertTokenizer, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import json
import random
import numpy as np
import torch.nn as nn
from utils import *
from models import *
import torch
import pickle
import copy
import pdb

fname = sys.argv[1]
print(fname)
op_fname = sys.argv[2]
print(op_fname)

obj = pickle.load(open(fname, 'rb'))
"""
>>> type(obj)
<class 'list'>
>>> type(obj[0])
<class 'dict'>
>>> obj[0].keys()
dict_keys(['input', 'source-score', 'iters'])
>>> type(obj[0]['iters'])
<class 'list'>
>>> type(obj[0]['iters'][0])
<class 'dict'>
>>> obj[0]['iters'][0].keys()
dict_keys(['idx', 'ip_iter', 'op'])
>>> obj[0]['iters'][0]['op']
>>> len(obj[0]['iters'])
20
"""

is_model = None
if 'model' in fname:
    is_model=True
elif 'baseline' in fname:
    is_model=False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sc_dir = "/path/to/best/bestmodel.tar" 

collate = ASCollater(vocab, Tokenizer(vocab), pad=True)
cnn_model = FluorescenceModel(len(vocab), 5, 1024, 0) 
device = torch.device('cuda:0')
cnn_model = cnn_model.to(device)
sd = torch.load(sc_dir)
cnn_model.load_state_dict(sd['model_state_dict'])
print('loaded the saved model from ', sc_dir)
cnn_model = cnn_model.eval()

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def test_step(model, batch):
    src, tgt, mask = batch
    src = src.to(device).float()
    tgt = tgt.to(device).float()
    mask = mask.to(device).float()
    output = model(src, mask)

    return output.detach().cpu(), tgt.detach().cpu()

header_str = "M A A D G Y L P D W L E D T L S E G I R Q W W K L K P G P P P P K P A E R H K D D S R G L V L P G Y K Y L G P F N G L D K G E P V N E A D A A A L E H D K A Y D R Q L D S G D N P Y L K Y N H A D A E F Q E R L K E D T S F G G N L G R A V F Q A K K R V L E P L G L V E E P V K T A P G K K R P V E H S P V E P D S S S G T G K A G Q Q P A R K R L N F G Q T G D A D S V P D P Q P L G Q P P A A P S G L G T N T M A T G S G A P M A D N N E G A D G V G N S S G N W H C D S T W M G D R V I T T S T R T W A L P T Y N N H L Y K Q I S S Q S G A S N D N H Y F G Y S T P W G Y F D F N R F H C H F S P R D W Q R L I "

wt = "N N N W G F R P K R L N F K L F N I Q V K E V T Q N D G T T T I A N N L T S T V Q V F T D S E Y Q L P Y V L G S A H Q G C L P P F P A D V F M V P Q Y G Y L T L N N G S Q A V G R S S F Y C L E Y F P S Q M L R T G N N F T F S Y T F E D V P F H S S Y A H S Q S L D R L M N P L I D Q Y L Y Y L S R T N T P S G T T T Q S R L Q F S Q A G A S D I R D Q S R N W L P G P C Y R Q Q R V S K T S A D N N N S E Y S W T G A T K Y H L N G R D S L V N P G P A M A S H K D D E E K F F P Q S G V L I F G K Q G S E K T N V D I E K V M I T D E E E I R T T N P V A T E Q Y G Q C V C S W E H Q G G S N Q Y Q A A T A D V N T Q G V L P G M V W Q D R D V Y L Q G P I W A K I P H T D G H F H P S P L M G G F G L K H P P P Q I L I K N T P V P A N P S T T F S A A K F A S F I T Q Y S T G Q V S V E I E W E L Q K E N S K R W N P E I Q Y T S N Y N K S V N V D F T V D T N G V Y S E P R P I G T R Y L T R N L"
val_ip = ["<inc> "+wt for i in range(50000)]

match_str = ['T', 'S', 'N', 'Y', 'N', 'K', 'S', 'V', 'N', 'V', 'D', 'F', 'T', 'V', 'D', 'T', 'N', 'G', 'V', 'Y', 'S', 'E', 'P', 'R', 'P', 'I', 'G', 'T', 'R', 'Y', 'L', 'T', 'R', 'N', 'L']

def filter_string(s):
    s = s.split()
    indices = find_sub_list(match_str, s)
    assert len(indices) == 1, print(s, match_str, indices)
    start, end = indices[0]
    return ' '.join(s[:(end+1)])


scorer_op = []
for item in obj:
    ip = ''.join((header_str + wt).split())
    iter_0 = [''.join((header_str+filter_string(text)).split()) for text in item['iters'][0]['op']]#_text']]
    iter_1 = [''.join((header_str+filter_string(text)).split()) for text in item['iters'][1]['op']]#_text']]
    iter_10 = [''.join((header_str+filter_string(text)).split()) for text in item['iters'][9]['op']]#_text']]
    iter_l = [''.join((header_str+filter_string(text)).split()) for text in item['iters'][-1]['op']]#_text']]

    temp = copy.deepcopy(item)
    temp['output_scores'] = {}
    for text_idx, texts in enumerate([[ip], iter_0, iter_1, iter_10, iter_l]): 
        batch_size = 30 
        test = pd.DataFrame.from_dict({'sequence':texts, 'target':[-1.5 for i in range(len(texts))], 'validation':[None for i in range(len(texts))], 'set':["test" for i in range(len(texts))]})
        test_set = SequenceDataset(test)
        test_iterator = DataLoader(test_set, collate_fn=collate, batch_size=batch_size)
        for i, batch in enumerate(test_iterator):
            # print(batch)
            output, tgt = test_step(cnn_model, batch)
            # print(output, tgt)
        assert i == 0

        flat_list = output 
        if text_idx == 0:
            temp['output_scores']['input'] = flat_list[0]
        elif text_idx == 1:
            temp['output_scores']['0'] = flat_list
        elif text_idx == 2:
            temp['output_scores']['1'] = flat_list
        elif text_idx == 3:
            temp['output_scores']['10'] = flat_list
        elif text_idx == 4:
            temp['output_scores']['last'] = flat_list

    scorer_op.append(temp)
    pickle.dump(scorer_op, open(op_fname, 'wb'))

