import pandas as pd
import numpy as np
import csv

# a_0 = pd.read_csv('ninju-preds-AA-grid.csv').to_numpy()
# a_1 = pd.read_csv('preds-alltrain-5.csv').to_numpy()
a_2 = pd.read_csv('preds-bert-bilstm.csv').to_numpy()
a_3 = pd.read_csv('preds-best.csv').to_numpy()
a_4 = pd.read_csv('predsPoolBertLin5.csv').to_numpy()

preds = np.zeros((4957,2))

print(preds.shape)

for i in range(4957):
    l = [0,0,0]
    for j in range(5):
        # l[a_0[i,0]] += 1
        # l[a_1[i,0]] += 1
        l[a_2[i,0]] += 1
        l[a_3[i,0]] += 1
        l[a_4[i,0]] += 1
    
    preds[i,1] = i
    if l[0] == max(l):
        preds[i,0] = 0
    elif l[2] == max(l):
        preds[i,0] = 2
    else:
        preds[i,0] = 1

preds = preds.astype(np.int32)     
print(preds)  

header = ["label", "id"]
with open("FINAL.csv", 'w', encoding='UTF8') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)

    for row in preds:
        csv_writer.writerow(row) 


