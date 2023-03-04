import pandas as pd
import sys
from transformers import T5Tokenizer, T5Model
import re
import torch
import json
from transformers import AlbertTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = AlbertTokenizer.from_pretrained("prot_t5_xl_uniref50")
model = AutoModelForSeq2SeqLM.from_pretrained("prot_t5_xl_uniref50").to(device)

df = pd.read_csv("ace2_250k.csv")
output_file = "mutating_from_wt.jsonl"

wt_seq = "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ"
wt = ' '.join(wt_seq)
ids = tokenizer.encode(wt, add_special_tokens=True, padding=True)
reserved_span = "NTNITEEN"
reserved_span_start = 32
constant_indexes = [i for i in range(reserved_span_start, reserved_span_start+len(reserved_span))]

mutable_indexes = [i for i in range(len(wt_seq)) if i not in constant_indexes]

op = []
for i in range(10000):
    mut_init = df['MT_seq'][i]
    temp = [1 for idx in mutable_indexes]
    ip_mask = torch.bernoulli(0.95 * torch.tensor(temp))

    mut = ""
    for i in range(len(wt_seq)):
        if i not in mutable_indexes:
            mut+=mut_init[i]+' ' 
            continue
        mask_idx = mutable_indexes.index(i)
        if ip_mask[mask_idx] == 0:
            mut+="[MASK] "
        else:
            mut+=mut_init[i]+' '

    model_input = tokenizer.batch_encode_plus([mut], add_special_tokens=True, padding=True)
    # print(model_input)

    # print(wt)
    # print(ids)

    # print(mut)

    embedding = model.generate(input_ids=torch.tensor(model_input['input_ids']).to(device), attention_mask=torch.tensor(model_input['attention_mask']).to(device), max_length = 128)
    output = tokenizer.batch_decode(embedding, skip_special_tokens=True)
    # print(output)
    item = {}
    item['wt'] = wt
    item['masked wt'] = mut
    # item['ip_mask'] = ip_mask
    item['output'] = output
    op.append(item)

    if len(op) % 500 == 0:
        with open(output_file, 'w') as f:
            for item in op:
                f.write(json.dumps(item)+'\n')
    # breakpoint()

with open(output_file, 'w') as f:
    for item in op:
        f.write(json.dumps(item)+'\n')
