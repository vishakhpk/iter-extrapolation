import pickle
import scipy.stats as st
import json
import random
import pdb
import copy
import sys

denoise_file = sys.argv[1]
print(denoise_file)
output_dir = sys.argv[2]
print(output_dir)
op = []
all_src = []
all_scores = []
all_lengths = []
c_inc = 0
c_dec = 0
obj = pickle.load(open(denoise_file, "rb"))
for k in range(len(obj['ip'])):
    d = {}
    src = obj['ip'][k]
    tgt = obj['op'][k][0]
    if src.strip() == tgt.strip():
        continue
    src = src.encode('ascii', 'ignore').decode('ascii')
    src = src.replace('\\n', ' ')
    src = src.replace('\\\"', ' ')
    src = src.replace('\t', ' ')
    tgt = tgt.encode('ascii', 'ignore').decode('ascii')
    tgt = tgt.replace('\\n', ' ')
    tgt = tgt.replace('\\\"', ' ')
    tgt = tgt.replace('\t', ' ')
    all_src.append(src)
    if float(obj['ip-score'][k]) < 1 or float(obj['ip-score'][k]) > 3:
        continue
    if float(obj['op-score'][k]) < 1 or float(obj['op-score'][k]) > 3:
        continue
    abs_diff = abs(float(obj['ip-score'][k]) - float(obj['op-score'][k]))
    if abs_diff < 0.5: # or abs_diff < 0.2:
        continue
    if float(obj['ip-score'][k]) < float(obj['op-score'][k]):
        tok = "<inc> "
        rev_tok = "<dec> "
        c_inc+=1
    else:
        tok = "<dec> "
        rev_tok = "<inc> "
        c_dec+=1
    d['translation'] = {}
    d['translation']['src'] = tok+src
    d['translation']['tgt'] = tgt
    d['translation']['ip-score'] = obj['ip-score'][k]
    d['translation']['op-score'] = obj['op-score'][k]
    all_lengths.append((len(src.split()), len(tgt.split())))
    all_scores.append(obj['ip-score'][k])
    op.append(copy.deepcopy(d))
    #print(d)
    d['translation'] = {}
    d['translation']['src'] = rev_tok+tgt
    d['translation']['tgt'] = src
    d['translation']['ip-score'] = obj['op-score'][k]
    d['translation']['op-score'] = obj['ip-score'][k]
    all_lengths.append((len(src), len(tgt)))
    all_scores.append(obj['op-score'][k])
    op.append(copy.deepcopy(d))
        #print(d)
        #pdb.set_trace()

print(len(op), c_inc, c_dec)         
random.shuffle(op)
train = op[:int(0.9*len(op))]
val = op[int(0.9*len(op)):]
print(len(train), len(val))
statistic, bins, binnumber = st.binned_statistic(all_scores, all_scores, statistic = 'count', bins = 10)
print("Score stats")
print([int(i) for i in statistic])
print([round(i, 3) for i in bins])
print(binnumber)

print("Length stats")
s_len = [i[0] for i in all_lengths]
t_len = [i[1] for i in all_lengths]
print(max(s_len), min(s_len), sum(s_len)/len(s_len))
print(max(t_len), min(t_len), sum(t_len)/len(t_len))
statistic, bins, binnumber = st.binned_statistic(s_len, s_len, statistic = 'count', bins = 10)
print(statistic, bins, binnumber)
#pdb.set_trace()
# exit(0)

with open(output_dir+"/train.json", "w") as f:
    for item in train:
        json.dump(item, f)
        f.write('\n')

with open(output_dir+"/val.json", "w") as f:
    for item in val:
        if len(item['translation']['tgt']) < 5:
            continue
        json.dump(item, f)
        f.write('\n')

