import sys
import scipy.stats as st
import json
import pandas as pd
import copy
import random

filename = sys.argv[1]
output_dir = sys.argv[2]
if output_dir[-1] != '/':
    output_dir += '/'
df = pd.read_json(filename, lines=True)
ip = df.to_dict(orient='records')
# print(ip[0].keys())
c_inc, c_dec = 0, 0
op = []
all_scores = []

random.shuffle(ip)

for loop_idx in range(1000000):
    src_idx = loop_idx % len(ip)
    c = 0
    src = ip[src_idx]
    for j in range(len(ip)):
        if c == 10:
            break
        tgt_idx = random.randint(0, len(ip)-1)
        tgt = ip[tgt_idx]
        abs_diff = abs(src['label'] - tgt['label'])
        if abs_diff > 2.5:
            continue
        if src['label'] > 0 or src['label'] < -5:
            continue
        if tgt['label'] > 0 or tgt['label'] > -5:
            continue

        if src['label'] < tgt['label']:
            tok = "<inc> "
            rev_tok = "<dec> "
            c_inc+=1
        else:
            tok = "<dec> "
            rev_tok = "<inc> "
            c_dec+=1

        d = {}
        d['translation'] = {}
        d['translation']['src'] = tok+' '.join(src['mutated text'])
        d['translation']['tgt'] = ' '.join(tgt['mutated text'])
        d['translation']['ip-score'] = src['label']
        d['translation']['op-score'] = tgt['label']
        all_scores.append(src['label'])
        op.append(copy.deepcopy(d))
        #print(d)
        d['translation'] = {}
        d['translation']['src'] = rev_tok+' '.join(tgt['mutated text'])
        d['translation']['tgt'] = ' '.join(src['mutated text'])
        d['translation']['ip-score'] = tgt['label']
        d['translation']['op-score'] = src['label']
        all_scores.append(tgt['label'])
        op.append(copy.deepcopy(d))

        c+=1

        # breakpoint()
    if len(op) > 1000000:
        break

    if len(op) % 50000 == 0:
        with open(output_dir+"checkpoint_gen.json", "w") as f:
            for item in op:
                json.dump(item, f)
                f.write('\n')


print(len(op), c_inc, c_dec)         
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

with open(output_dir+"train_gen.json", "w") as f:
    for item in train:
        json.dump(item, f)
        f.write('\n')

with open(output_dir+"val_gen.json", "w") as f:
    for item in val:
        json.dump(item, f)
        f.write('\n')

