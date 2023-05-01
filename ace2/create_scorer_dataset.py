import sys
import scipy.stats as st
import json
import pandas as pd
import copy
import random

filename = sys.argv[1]
output_dir = sys.argv[2]
df = pd.read_csv(filename)
ip = df.to_dict(orient='records')
print(ip[0].keys())
c_inc, c_dec = 0, 0
random.shuffle(ip)

op = []
all_scores = []

for src_idx in range(len(ip)):
    c = 0
    src = ip[src_idx]
    if src['ddG'] < -4 or src['ddG'] > 3:
        continue

    op_item = {}
    op_item['label'] = src['ddG']
    op_item['text'] =  ' '.join(src['MT_seq'])
    # op_item['original'] = src
    op.append(op_item)
    all_scores.append(op_item['label'])

print(len(op))         
# print(len(list(set([item['translation']['src'][5:] for item in op]))))
random.shuffle(op)
train = op[:int(0.9*len(op))]
val = op[int(0.9*len(op)):]
print(len(train), len(val))
statistic, bins, binnumber = st.binned_statistic(all_scores, all_scores, statistic = 'count', bins = 10)
print("Score stats")
print([int(i) for i in statistic])
print([round(i, 3) for i in bins])
print(binnumber)

if output_dir[-1] != '/':
    output_dir += '/'
with open("train_disc.json", "w") as f:
    for item in train:
        json.dump(item, f)
        f.write('\n')

with open("val_disc.json", "w") as f:
    for item in val:
        json.dump(item, f)
        f.write('\n')

