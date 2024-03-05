import sys
from transformers import AlbertTokenizer, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import json
import pickle
import random
import numpy as np
import torch
import pdb
import torch.nn as nn
np.random.seed(42)

model_dir = sys.argv[1]
model_type = sys.argv[2]
op_dir = sys.argv[3]
op_name = sys.argv[4]

print("-"*40)
print("Beginning eval")
print("-"*40)
print(model_dir, model_type)

config = AutoConfig.from_pretrained(model_dir)
tokenizer = AlbertTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)#.to('cuda')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


header_str = "M A A D G Y L P D W L E D T L S E G I R Q W W K L K P G P P P P K P A E R H K D D S R G L V L P G Y K Y L G P F N G L D K G E P V N E A D A A A L E H D K A Y D R Q L D S G D N P Y L K Y N H A D A E F Q E R L K E D T S F G G N L G R A V F Q A K K R V L E P L G L V E E P V K T A P G K K R P V E H S P V E P D S S S G T G K A G Q Q P A R K R L N F G Q T G D A D S V P D P Q P L G Q P P A A P S G L G T N T M A T G S G A P M A D N N E G A D G V G N S S G N W H C D S T W M G D R V I T T S T R T W A L P T Y N N H L Y K Q I S S Q S G A S N D N H Y F G Y S T P W G Y F D F N R F H C H F S P R D W Q R L I "

wt = "N N N W G F R P K R L N F K L F N I Q V K E V T Q N D G T T T I A N N L T S T V Q V F T D S E Y Q L P Y V L G S A H Q G C L P P F P A D V F M V P Q Y G Y L T L N N G S Q A V G R S S F Y C L E Y F P S Q M L R T G N N F T F S Y T F E D V P F H S S Y A H S Q S L D R L M N P L I D Q Y L Y Y L S R T N T P S G T T T Q S R L Q F S Q A G A S D I R D Q S R N W L P G P C Y R Q Q R V S K T S A D N N N S E Y S W T G A T K Y H L N G R D S L V N P G P A M A S H K D D E E K F F P Q S G V L I F G K Q G S E K T N V D I E K V M I T D E E E I R T T N P V A T E Q Y G Q C V C S W E H Q G G S N Q Y Q A A T A D V N T Q G V L P G M V W Q D R D V Y L Q G P I W A K I P H T D G H F H P S P L M G G F G L K H P P P Q I L I K N T P V P A N P S T T F S A A K F A S F I T Q Y S T G Q V S V E I E W E L Q K E N S K R W N P E I Q Y T S N Y N K S V N V D F T V D T N G V Y S E P R P I G T R Y L T R N L"
val_ip = ["<inc> "+wt for i in range(50000)]

print(len(val_ip))

success = 0
total = 0
logs = []
dist_size = 50
batch_size = 4
for item in val_ip:
    print(len(logs))
    ip = item 
    assert ip[0] == '<' and ip[4] == '>'
    total += 1
    res = {}
    res['input'] = ip
    res['source-score'] = 0 
    res['iters'] = []
    iteration = 0
    ip_text = ip
    ip_iter = [ip_text for i in range(batch_size)]
    tok = "<dec> "
    while iteration < 10:
        end_iter = False
        op = []
        batch = tokenizer(ip_iter, return_tensors="pt").input_ids.to(device)
        translated = model.generate(batch, do_sample = True, top_k = 10, num_return_sequences = 1, temperature = 0.7, early_stopping=True)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens = True)
        op_iter = [tgt_text[i] for i in range(batch_size)]
        op.extend(op_iter)
        res['iters'].append({'idx':iteration, 'ip_iter':ip_iter, 'op':op})
        iteration += 1
        ip_iter = [tok+item for item in op]
            
    logs.append(res)
    if len(logs) % 100 == 0:
        pickle.dump(logs, open(op_dir+op_name+"-"+model_type+".pkl", "wb"))

pickle.dump(logs, open(op_dir+op_name+"-"+model_type+".pkl", "wb"))
