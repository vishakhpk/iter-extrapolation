import sys
import json
import pandas as pd
import csv
import pickle

ip_file = sys.argv[1]
op_file = sys.argv[2]
obj = pickle.load(open(ip_file, 'rb'))
print(ip_file, type(obj))
# breakpoint()

c = 0
op = []
for item in obj:
    op_item = {}
    op_item['WT_seq'] = item['input'][6:].replace(" ", "")
    assert len(op_item['WT_seq']) == 83, print(op_item['WT_seq'])
    op_item['MT_seq'] = item['iters'][0]['op'][0].replace(" ", "")
    op_item['PDB'] = "template2"
    op_item['Chain'] = "A"
    op_item['Start_index'] = 19
    if op_item['WT_seq'] == op_item['MT_seq']:
        c+=1
        continue
    op.append(op_item)

df = pd.DataFrame(op)
num_rows = 25
df_dict = {n: df.iloc[n:n+num_rows, :] for n in range(0, len(df), num_rows)}
print(c, df.shape, df_dict[0].shape)
for i, d in enumerate(df_dict):
    df.to_csv(str(i)+"-"+op_file, sep='\t')
