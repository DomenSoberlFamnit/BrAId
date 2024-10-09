import numpy as np

def run(dir_braid, num_training_samples, min_testing_samples):
    print("Loading data_id.npy")
    data_id = np.load(f'{dir_braid}data_id.npy')
    
    print("Loading data_x.npy")
    data_x = np.load(f'{dir_braid}data_x.npy')

    print("Loading data_y.npy")
    data_y = np.load(f'{dir_braid}data_y.npy')

    # Separate indices according to classes.
    distribution = []
    for _ in range(len(data_y[0])):
        distribution.append([])

    for i in range(len(data_y)):
        y = data_y[i]
        cls = np.argmax(y)
        distribution[cls].append(i)
    
    # Shuffle indices within each class.
    distribution_np = []
    for samples in distribution:
        samples_np = np.array(samples, dtype=np.uint32)
        np.random.shuffle(samples_np)
        distribution_np.append(samples_np)
    distribution = distribution_np

    # Select samples
    training_indices = np.array([], dtype=np.uint32)
    testing_indices = np.array([], dtype=np.uint32)
    for samples in distribution:
        if samples.size >= num_training_samples + min_testing_samples:
            training_indices = np.concatenate((training_indices, samples[0:num_training_samples]))
            testing_indices = np.concatenate((testing_indices, samples[num_training_samples:]))
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
    
    # Shuffle the training indices
    np.random.shuffle(training_indices)

    # Save samples
    training_id = data_id[training_indices]
    training_x = data_x[training_indices]
    training_y = data_y[training_indices]
    testing_id = data_id[testing_indices]
    testing_x = data_x[testing_indices]
    testing_y = data_y[testing_indices]

    print('Saving training_id.npy')
    np.save(f'{dir_braid}training_id.npy', training_id)

    print('Saving training_x.npy')
    np.save(f'{dir_braid}training_x.npy', training_x)
    
    print('Saving training_y.npy')
    np.save(f'{dir_braid}training_y.npy', training_y)

    print('Saving testing_id.npy')
    np.save(f'{dir_braid}testing_id.npy', testing_id)

    print('Saving testing_x.npy')
    np.save(f'{dir_braid}testing_x.npy', testing_x)
    
    print('Saving testing_y.npy')
    np.save(f'{dir_braid}testing_y.npy', testing_y)

    
    


