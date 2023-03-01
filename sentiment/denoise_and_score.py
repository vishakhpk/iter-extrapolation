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
from transformers import BartForConditionalGeneration, BartTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import sys 

ip = []
dataset_name = sys.argv[1]
op_dir = sys.argv[2]
if op_dir[-1]!="/":
    op_dir = op_dir+'/'
model_dir = sys.argv[3]
input_idx = sys.argv[4]

print("Starting generation")
data = pickle.load(open(op_dir+dataset_name+"-denoise-ip-lists"+input_idx+".pkl", "rb"))

op = data['op']
ip = data['ip']
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', cache_dir = "./hf-cache")
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', cache_dir = "./hf-cache")
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(torch_device)
try:
    final = pickle.load(open(op_dir+dataset_name+"-denoise-final-large"+input_idx+".pkl", "rb")) # []
except:
    final = {}

disc_tokenizer = AutoTokenizer.from_pretrained(model_dir)
disc_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to('cuda')

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

n_seq = 1
batch_size = 8
for i in range(0, len(op), batch_size):
    if i in final.keys(): 
        continue
     
    batch = tokenizer(op[i:i+batch_size],truncation=True,padding='longest',max_length=512, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=512,num_beams=5, num_return_sequences=n_seq, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    disc_ip = []
    for j in range(batch_size):
        assert i+j not in final.keys()
        disc_ip.append(ip[i+j]+" </s> "+tgt_text[j])

    encoded_input = disc_tokenizer(disc_ip, padding=True, truncation=True, max_length = 512, return_tensors='pt')
    output = disc_model(**encoded_input.to('cuda'))
    t = output['logits'].tolist()
    flat_list = [item for sublist in t for item in sublist]

    for j in range(batch_size):
        assert i+j not in final.keys()
        final[i+j] = {}
        final[i+j]['ip'] = ip[i+j]
        final[i+j]['masked'] = op[i+j]
        final[i+j]['op'] = tgt_text[j]
        final[i+j]['disc-score'] = flat_list[j]

    if len(final.keys()) % 15000 == 0:
        print(len(final.keys()))
        pickle.dump(final, open(op_dir+dataset_name+"-denoise-final-large"+input_idx+".pkl", "wb"))

pickle.dump(final, open(op_dir+dataset_name+"-denoise-final-large"+input_idx+".pkl", "wb"))
