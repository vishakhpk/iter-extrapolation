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

wt = "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ"
val_ip = ["<dec> "+' '.join(wt) for i in range(50000)]

print(len(val_ip))

success = 0
total = 0
logs = []
dist_size = 50
batch_size = 4
for item in val_ip:
    print(len(logs))
    ip = item #['translation']['src']
    assert ip[0] == '<' and ip[4] == '>'
    total += 1
    res = {}
    res['input'] = ip
    res['source-score'] = 0 #item['translation']['ip-score']
    res['iters'] = []
    iteration = 0
    ip_text = ip
    ip_iter = [ip_text for i in range(batch_size)]
    tok = "<dec> "
    while iteration < 20:
        end_iter = False
        op = []
        batch = tokenizer(ip_iter, return_tensors="pt").input_ids.to(device)
        translated = model.generate(batch, do_sample = True, top_k = 10, num_return_sequences = 1, temperature = 0.7, early_stopping=True)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens = True)
        op_iter = [tgt_text[i][:165] for i in range(batch_size)]
        op.extend(op_iter)
        res['iters'].append({'idx':iteration, 'ip_iter':ip_iter, 'op':op})
        iteration += 1
        ip_iter = [tok+item for item in op]
            
    logs.append(res)
    if len(logs) % 100 == 0:
        pickle.dump(logs, open(op_dir+op_name+"-"+model_type+".pkl", "wb"))

pickle.dump(logs, open(op_dir+op_name+"-"+model_type+".pkl", "wb"))
