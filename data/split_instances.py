import os
import json
import numpy as np

def run(dir_braid, num_training_samples, min_testing_samples, max_testing_samples, use_predefined_ids=False):
    print("Loading data_id.npy")
    data_id = np.load(f'{dir_braid}data/data_id.npy')
    
    print("Loading data_x.npy")
    data_x = np.load(f'{dir_braid}data/data_x.npy')

    print("Loading data_y.npy")
    data_y = np.load(f'{dir_braid}data/data_y.npy')

    print("Loading group index")
    with open(f'{dir_braid}group_index.json') as file:
        groups_names = list(json.load(file).keys())

    # Load the predefined testing ids.
    if use_predefined_ids:
        if os.path.exists('../metadata/testing_ids.json'):
            print("Loading the predefined testings IDs.")
            with open('../metadata/testing_ids.json') as file:
                predefined_ids = json.load(file)
            print(f'Found {len(predefined_ids)} predefined testing IDs.')
        else:
            print("No predefined testing ids.")
    else:
        print("Not using predefined testing ids.")
        predefined_ids = []

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

    # Concatenate the two distributions.
    distribution_concat = []
    for (samples, samples_test) in zip(distribution, distribution_test):
        if len(samples) <= num_training_samples:
            distribution_concat.append(np.concatenate((samples, samples_test)))
        else:
            distribution_concat.append(np.concatenate((samples[0:num_training_samples], samples_test)))
    distribution = distribution_concat

    print("Computing the split.")

    # Select samples.
    training_indices = np.array([], dtype=np.uint32)
    testing_indices = np.array([], dtype=np.uint32)
    for samples in distribution:
        if samples.size >= num_training_samples + min_testing_samples:
            training_indices = np.concatenate((training_indices, samples[0:num_training_samples]))
            if samples.size <= num_training_samples + max_testing_samples:
                testing_indices = np.concatenate((testing_indices, samples[num_training_samples:]))
            else:
                testing_indices = np.concatenate((testing_indices, samples[num_training_samples:(num_training_samples + max_testing_samples)]))
        else:
            # Oversample
            split_idx = samples.size - min_testing_samples
            
            #training_indices = np.concatenate((training_indices, samples[0:split_idx]))
            oversample_set = np.array([], dtype=np.uint32)
            while oversample_set.size < num_training_samples:
                oversample_set = np.concatenate((oversample_set, samples[0:split_idx]))
            oversample_set = oversample_set[0:num_training_samples]
            
            training_indices = np.concatenate((training_indices, oversample_set)) 
            testing_indices = np.concatenate((testing_indices, samples[split_idx:]))
    
    # Shuffle the training indices.
    np.random.shuffle(training_indices)

    # Filter the samples.
    training_id = data_id[training_indices]
    training_x = data_x[training_indices]
    training_y = data_y[training_indices]
    testing_id = data_id[testing_indices]
    testing_x = data_x[testing_indices]
    testing_y = data_y[testing_indices]

    # Check the testing instances.
    distribution = {}
    cnt_testing_predefined = 0
    for (id, y) in zip(testing_id, testing_y):
        group_name = groups_names[np.argmax(y)]
        if str(id) in predefined_ids:
            if group_name not in distribution:
                distribution[group_name] = []
            distribution[group_name].append(id)
            cnt_testing_predefined += 1
    
    print(f'We have {cnt_testing_predefined}/{len(testing_id)} predefined testing IDs.')
    for group_name in distribution:
        print(f'{group_name}: {len(distribution[group_name])}')

    print('Saving training_id.npy')
    np.save(f'{dir_braid}data/training_id.npy', training_id)

    print('Saving training_x.npy')
    np.save(f'{dir_braid}data/training_x.npy', training_x)
    
    print('Saving training_y.npy')
    np.save(f'{dir_braid}data/training_y.npy', training_y)

    print('Saving testing_id.npy')
    np.save(f'{dir_braid}data/testing_id.npy', testing_id)

    print('Saving testing_x.npy')
    np.save(f'{dir_braid}data/testing_x.npy', testing_x)
    
    print('Saving testing_y.npy')
    np.save(f'{dir_braid}data/testing_y.npy', testing_y)

    
    


