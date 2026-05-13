groups = ["113", "1211", "122", "11", "22", "111", "112", "1112", "12", "1111", "123", "1212", "1222"]

precisions = {}

#f = open('stat-vgg16.csv', 'r')
f = open('stat-resnet.csv', 'r')

raised_cnt = 0

for line in f:
    cols = line.strip().split(',')
    camera = cols[2]
    road = cols[3]
    raised = cols[5]

    tp = cols[11]
    fp = cols[13]

    if raised == '1':
        raised_cnt += 1

    if camera not in precisions:
        precisions[camera] = {'tp':0, 'fp':0}
    
    if tp == '1':
        precisions[camera]['tp'] += 1
    if fp == '1':
        precisions[camera]['fp'] += 1
    
for group in groups:
    tp = precisions[group]['tp']
    fp = precisions[group]['fp']
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f'{group:<5}{tp/(tp+fp)}')
    else:
        print(f'{group:<5}-')

f.close()