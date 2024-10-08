import numpy as np
import json

def run(dir_braid):
    print("Loading group_index.json")
    file = open(f'{dir_braid}group_index.json')
    group_index = json.load(file)
    file.close()

    print("Loading training_y.npy")
    training_y = np.load(f'{dir_braid}training_y.npy')

    print("Loading testing_y.npy")
    testing_y = np.load(f'{dir_braid}testing_y.npy')

    group_index = dict(sorted(group_index.items(), key=lambda item: item[1], reverse=False))
    groups = list(group_index.keys())

    count_training = {}
    for y in training_y:
        index = np.argmax(y)
        group_name = groups[index]
        if not group_name in count_training:
            count_training[group_name] = 0
        count_training[group_name] += 1
    
    count_testing = {}
    for y in testing_y:
        index = np.argmax(y)
        group_name = groups[index]
        if not group_name in count_testing:
            count_testing[group_name] = 0
        count_testing[group_name] += 1
    
    for group_name in count_training:
        print(f'Group {group_name} has {count_training[group_name]} training and {count_testing[group_name]} testing instances.')

    