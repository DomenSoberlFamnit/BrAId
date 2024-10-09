import numpy as np
import json

def run(dir_braid):
    print("Loading group_index.json")
    file = open(f'{dir_braid}group_index.json')
    group_index = json.load(file)
    file.close()

    print("Loading training_id.npy")
    training_id = np.load(f'{dir_braid}training_id.npy')

    print("Loading testing_id.npy")
    testing_id = np.load(f'{dir_braid}testing_id.npy')

    print("Loading training_y.npy")
    training_y = np.load(f'{dir_braid}training_y.npy')

    print("Loading testing_y.npy")
    testing_y = np.load(f'{dir_braid}testing_y.npy')

    group_index = dict(sorted(group_index.items(), key=lambda item: item[1], reverse=False))
    groups = list(group_index.keys())

    # Count how many training instances each group has.
    print("Checking training instances.")
    count_training = {}
    for y in training_y:
        index = np.argmax(y)
        group_name = groups[index]
        if not group_name in count_training:
            count_training[group_name] = 0
        count_training[group_name] += 1
    
    # Count how many testing instances each group has.
    print("Checking testing instances.")
    count_testing = {}
    for y in testing_y:
        index = np.argmax(y)
        group_name = groups[index]
        if not group_name in count_testing:
            count_testing[group_name] = 0
        count_testing[group_name] += 1
    
    # Count how many distinct training instances each group has.
    print("Checking duplicates instances.")
    count_distinct = {}
    for (y, id) in zip(training_y, training_id):
        index = np.argmax(y)
        group_name = groups[index]
        if not group_name in count_distinct:
            count_distinct[group_name] = []
        if id not in count_distinct[group_name]:
            count_distinct[group_name].append(id)

    # Check if training and testing sets are disjunct.
    print("Checking set intersection.")
    count_intersecting = 0
    for id in training_id:
        if id in testing_id:
            count_intersecting += 1

    for group_name in groups:
        print(f'{group_name}: {count_training[group_name]} training / {count_testing[group_name]} testing / {len(count_distinct[group_name])} distinct training.')
    print(f'The number of instances intersecting the training and the testing set: {count_intersecting}')