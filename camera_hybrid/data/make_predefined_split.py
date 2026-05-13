import os
import json
import numpy as np

dir_braid = '/home/hicup/disk/braid/'


def split_instances(dir_braid, num_training_samples, experiment):
    print("Loading data_id.npy")
    data_id = np.load(f'{dir_braid}data/data_id.npy')

    print("Loading data_x.npy")
    data_x = np.load(f'{dir_braid}data/data_x.npy')

    print("Loading data_y.npy")
    data_y = np.load(f'{dir_braid}data/data_y.npy')

    print("Loading group index")
    with open(f'{dir_braid}group_index.json') as file:
        group_index = json.load(file)
        groups_names = list(group_index.keys())

    # Load the predefined testing ids.
    if os.path.exists('../metadata/testing_ids.json'):
        print("Loading the predefined testings IDs.")
        with open('../metadata/testing_ids.json') as file:
            testing_ids = json.load(file)[str(experiment)]
        print(f'Found {len(testing_ids)} predefined testing IDs.')
    else:
        print("Not using predefined testing ids.")
        testing_ids = []

    assert len(testing_ids) > 0

    # Create ID index.
    id_index = {}
    cnt = 0
    for id in data_id:
        id_index[str(id)] = cnt
        cnt += 1

    # Correct the y-vectors.
    predefined_ids = {}
    for instance in testing_ids:
        id = instance['ID']
        groups = instance['class']
        predefined_ids[id] = groups
        idx = id_index[id]
        vector = np.zeros(len(groups_names))
        vector[group_index[groups]] = 1
        data_y[idx] = vector

    # Separate indices according to classes.
    distribution = []
    distribution_test = []
    for _ in range(len(data_y[0])):
        distribution.append([])
        distribution_test.append([])

    for i in range(len(data_y)):
        id = data_id[i]
        y = data_y[i]
        cls = np.argmax(y)
        if str(id) in predefined_ids:
            distribution_test[cls].append(i)
        else:
            distribution[cls].append(i)

    # Print out the statistics.
    print("Distribution of the predefined testing IDs:")
    for (group_name, ids) in zip(groups_names, distribution_test):
        print(f'{group_name}: {len(ids)}')
    print("Distribution of the rest of IDs:")
    for (group_name, ids) in zip(groups_names, distribution):
        print(f'{group_name}: {len(ids)}')

    # Shuffle indices within each class.
    distribution_np = []
    for samples in distribution:
        samples_np = np.array(samples, dtype=np.uint32)
        np.random.shuffle(samples_np)
        distribution_np.append(samples_np)
    distribution = distribution_np

    distribution_test_np = []
    for samples in distribution_test:
        samples_np = np.array(samples, dtype=np.uint32)
        np.random.shuffle(samples_np)
        distribution_test_np.append(samples_np)
    distribution_test = distribution_test_np

    print("Selecting the training instances.")

    training_indices = np.array([], dtype=np.uint32)
    for samples in distribution:
        if len(samples) >= num_training_samples:
            training_indices = np.concatenate(
                (training_indices, samples[0:num_training_samples]))
        else:
            # Oversample
            oversample_set = np.array([], dtype=np.uint32)
            while oversample_set.size < num_training_samples:
                oversample_set = np.concatenate((oversample_set, samples))
            training_indices = np.concatenate(
                (training_indices, oversample_set[0:num_training_samples]))

    # Select samples.
    testing_indices = np.array([], dtype=np.uint32)
    for samples in distribution_test:
        testing_indices = np.concatenate((testing_indices, samples))

    # Shuffle the training and the testing indices.
    np.random.shuffle(training_indices)
    np.random.shuffle(testing_indices)

    # Filter the samples.
    training_id = data_id[training_indices]
    training_x = data_x[training_indices]
    training_y = data_y[training_indices]
    testing_id = data_id[testing_indices]
    testing_x = data_x[testing_indices]
    testing_y = data_y[testing_indices]

    dir_data = f'{dir_braid}data{experiment}/'
    if not os.path.exists(dir_data):
        os.mkdir(dir_data)

    print('Saving training_id.npy')
    np.save(f'{dir_data}training_id.npy', training_id)

    print('Saving training_x.npy')
    np.save(f'{dir_data}training_x.npy', training_x)

    print('Saving training_y.npy')
    np.save(f'{dir_data}training_y.npy', training_y)

    print('Saving testing_id.npy')
    np.save(f'{dir_data}testing_id.npy', testing_id)

    print('Saving testing_x.npy')
    np.save(f'{dir_data}testing_x.npy', testing_x)

    print('Saving testing_y.npy')
    np.save(f'{dir_data}testing_y.npy', testing_y)


def check_instances(dir_braid, experiment):
    print("Loading group_index.json")
    file = open(f'{dir_braid}group_index.json')
    group_index = json.load(file)
    file.close()

    dir_data = f'{dir_braid}data{experiment}/'

    print("Loading training_id.npy")
    training_id = np.load(f'{dir_data}training_id.npy')

    print("Loading testing_id.npy")
    testing_id = np.load(f'{dir_data}testing_id.npy')

    print("Loading training_y.npy")
    training_y = np.load(f'{dir_data}training_y.npy')

    print("Loading testing_y.npy")
    testing_y = np.load(f'{dir_data}testing_y.npy')

    group_index = dict(
        sorted(group_index.items(), key=lambda item: item[1], reverse=False))
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
        print(f'{group_name}: {count_training[group_name]} training / {
              count_testing[group_name]} testing / {len(count_distinct[group_name])} distinct training.')
    print(f'The number of instances intersecting the training and the testing set: {
          count_intersecting}')


for i in range(1, 11):
    print(f'Making the split number {i}.')
    split_instances(dir_braid, 5000, i)
    check_instances(dir_braid, i)
