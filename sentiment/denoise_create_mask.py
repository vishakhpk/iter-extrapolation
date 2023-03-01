import os
import pdb
import pickle
import torch
import spacy
import random
nlp = spacy.load("en_core_web_sm")
import scipy.stats as sct
import datasets
from nltk.tokenize import sent_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
import sys 

def truncated_poisson(mu=6, max_value=8, size=1):
    temp_size = size
    while True:
        temp_size *= 2
        temp = sct.poisson.rvs(mu, size=temp_size)
        truncated = temp[temp <= max_value]
        if len(truncated) >= size:
            return truncated[:size]

ip = []
dataset_name = sys.argv[1]
op_dir = sys.argv[2]
if op_dir[-1]!="/":
    op_dir = op_dir+'/'

if len(sys.argv) >= 3:
    op_index = sys.argv[3]
else:
    op_index = ''

if dataset_name == "yelp":
    TEXT_COL, LABEL_COL = 'text', 'truth'
    colnames=[LABEL_COL, TEXT_COL]
    df = pd.read_csv("../yelp_data/train.csv", header=None, names=colnames)
    df[LABEL_COL] = df[LABEL_COL].astype(float) 
    data = {}
    data['train'] = datasets.Dataset.from_pandas(df)
    df = pd.read_csv("../yelp_data/test.csv", header=None, names=colnames)
    df[LABEL_COL] = df[LABEL_COL].astype(float) 
    data['dev'] = datasets.Dataset.from_pandas(df)

elif dataset_name == "sst5":
    TEXT_COL, LABEL_COL = 'text', 'truth'
    colnames=[LABEL_COL, TEXT_COL]
    df = pd.read_csv("../sst_data/sst_train.txt", sep='\t', header=None, names=colnames)
    df[LABEL_COL] = df[LABEL_COL].str.replace('__label__', '')
    df[LABEL_COL] = df[LABEL_COL].astype(float) 
    df[TEXT_COL] = df[TEXT_COL].str.replace("`", "'") # handle T5Tokenizer's inability to tokenize `, tokenizes it as <unk>
    # exit(0)
    data = {}
    data['train'] = datasets.Dataset.from_pandas(df)
    print(min(df[LABEL_COL]), max(df[LABEL_COL]), data['train'].shape)
    df = pd.read_csv("../sst_data/sst_dev.txt", sep='\t', header=None, names=colnames)
    df[LABEL_COL] = df[LABEL_COL].str.replace('__label__', '')
    df[LABEL_COL] = df[LABEL_COL].astype(float) 
    df[TEXT_COL] = df[TEXT_COL].str.replace("`", "'") # handle T5Tokenizer's inability to tokenize `, tokenizes it as <unk>
    # exit(0)
    data['dev'] = datasets.Dataset.from_pandas(df)
    print(min(df[LABEL_COL]), max(df[LABEL_COL]), data['dev'].shape)


data['train'] = data['train'].shuffle(seed = 42)

print(len(ip))
print(data)
print(data['train'])
print(data['train'].shape)
print(data['train'][0])
print(len(data['train']))#$.length)

op = []
remove = []
for i in range(len(data['train'])):
    ip.append(data['train'][i]['text'])
print(len(ip))

obj = {} 
obj['op'] = []
pickle.dump(obj, open(op_dir+dataset_name+"-denoise-ip-lists"+op_index+".pkl", "wb"))

for tot_idx in range(100000):
    idx = tot_idx % len(ip) 
    sentence = ip[idx]
    if idx % 1000 == 0:
        print(idx)
    try:
        doc = nlp(sentence)
    except:
        print("Failed ", sentence, idx)
        # remove.append(idx)
        continue
    temp = [1 for token in doc]
    tokens = [token for token in doc]
    ip_mask = torch.bernoulli(0.65 * torch.tensor(temp))
    ctr = 0
    sent = ""
    c = 0
    i = 0
    prev = ""
    while i < len(tokens):
        if ip_mask[c] == 0 and prev!="<mask> " and len(tokens[i]) > 1:
            length = truncated_poisson()
            sent+="<mask> "
            prev = "<mask> "
            c+=length[0]
            i+=length[0]
        else:
            sent+=tokens[i].text + " "
            prev = tokens[i].text + " "
            c+=1
            i+=1
    op.append(sent)
    if len(op) % 5000 == 0:
        obj['ip'] = ip
        obj['op'] = op
        pickle.dump(obj, open(op_dir+dataset_name+"-denoise-ip-lists"+op_index+".pkl", "wb"))


for i in range(5):
    print(ip[i], op[i])


obj = {}
obj['ip'] = ip
obj['op'] = op
pickle.dump(obj, open(op_dir+dataset_name+"-denoise-ip-lists"+op_index+".pkl", "wb"))
