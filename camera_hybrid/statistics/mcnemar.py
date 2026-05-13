import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix


method_1 = []
method_2 = []

filename1 = 'stat-resnet.csv'
filename2 = 'stat-resnet-noaug.csv'
print(filename1, filename2)

f1 = open(filename1, 'r')
f2 = open(filename2, 'r')

raised_cnt = 0

id_check = []

for line in f1:
    cols = line.strip().split(',')
    id = cols[0]
    siwim = cols[1]
    nn = cols[8]
    camera = cols[2]
    road = cols[3]
    raised = cols[5]
    agree = cols[10]

    id_check.append(id)
    #method_1.append(1 if nn == camera else 0)
    method_1.append(1 if cols[9] == '1' else 0)

idx = 0
for line in f2:
    cols = line.strip().split(',')
    id = cols[0]
    siwim = cols[1]
    nn = cols[8]
    camera = cols[2]
    road = cols[3]
    raised = cols[5]
    agree = cols[10]

    if id != id_check[idx]:
        print(f'ID missmatch: {id}, {id_check[idx]}')
        quit()
    idx += 1

    method_2.append(1 if nn == camera else 0)

f1.close()
f2.close()

table = confusion_matrix(method_1, method_2)
print(table)

result = mcnemar(table, exact=False, correction=True)
#print(result)
print(f"chi2 = {result.statistic:.4f}, p = {result.pvalue:.4f}")

if result.pvalue < 0.05:
    print("Significant difference between the two methods.")
else:
    print("No significant difference between the two methods.")
