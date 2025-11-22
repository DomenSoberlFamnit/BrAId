import numpy as np

groups = ["113", "1211", "122", "11", "22", "111", "112", "1112", "12", "1111", "123", "1212", "1222"]

def read_ca(filename):
    f = open(f'../results/{filename}', 'r')

    data = {}
    cnt = 0
    for line in f:
        cols = line.strip().split(',')
        truth = cols[8]
        nn_ca = cols[9]

        if truth not in data:
            data[truth] = {'value':0, 'count':0}
        
        if nn_ca == '1':
            data[truth]['value'] += 1
        data[truth]['count'] += 1
        
        cnt += 1
    f.close()

    return data, cnt

filename = 'stat-mobilenet.csv'

print(filename)
data, cnt = read_ca(filename)

for group in groups:
    if group in data:
        value = data[group]['value']
        count = data[group]['count']
        print(f'({group},{(100.0*value/count):.4f})', end=' ')
    else:
        print(f'({group},0.0000)', end=' ')
print()