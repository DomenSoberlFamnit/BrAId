import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, MobileNetV3Small, ResNet101V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import json

dir_braid = '/home/hicup/disk/braid/'
dir_models = f'{dir_braid}models/'
dir_results = f'{dir_braid}results/'

architectures = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'DenseNet121': DenseNet121,
    'MobileNetV3Small': MobileNetV3Small,
    'ResNet101V2': ResNet101V2
}

epochs = 25
sample_count = 0

def build_model(architecture, class_count):
    model = architecture(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def slot_indices(gpu_capacity, set_size):
    indices = []
    idx_from, idx_to = 0, 0
    while idx_to < set_size:
        idx_to += gpu_capacity
        if idx_to > set_size:
            idx_to = set_size
        indices.append((idx_from, idx_to, (idx_to - idx_from) / set_size))
        idx_from = idx_to
    return indices

def test(model, test_x, test_y, gpu_capacity=5000):
    correct = 0

    for (idx_from, idx_to, _) in slot_indices(gpu_capacity, len(test_x)):
        predictions = model.predict(test_x[idx_from:idx_to])
        for (prediction, true_y) in zip(predictions, test_y[idx_from:idx_to]):
            if np.argmax(prediction) == np.argmax(true_y):
                correct += 1
    
    return correct / len(test_x)

def train(model, name, epoch, train_x, train_y, testing_x, testing_y, gpu_capacity=5000):
    global sample_count

    for (idx_from, idx_to, _) in slot_indices(gpu_capacity, len(train_x)):
        time_start = timer()
        history = model.fit(x=train_x[idx_from:idx_to], y=train_y[idx_from:idx_to], batch_size=32, epochs=1, validation_split=0, shuffle=False)
        time_end = timer()
        
        time_ms = 1000 * (time_end - time_start) / (idx_to - idx_from)
        sample_count += idx_to - idx_from
        
        loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]

        test_accuracy = test(model, testing_x, testing_y)

        fname = f'{dir_results}{name}/training.txt'
        f = open(fname, 'a')
        f.write(f'{epoch + 1}, {sample_count}, {loss}, {train_accuracy}, {test_accuracy}, {time_ms}\n')
        f.close()

def process_model(name, architecture, group_index, training_x, training_y, testing_x, testing_y):
    global sample_count

    print(f'Building model {name}.')
    model = build_model(architecture, len(group_index))

    sample_count = 0
    for epoch in range(epochs):
        print(f'Training epoch {epoch + 1}/{epochs}.')
        train(model, name, epoch, training_x, training_y, testing_x, testing_y)

    print(f'Saving the model {name}.')
    model.save(f'{dir_models}{name}.keras')

    tf.keras.backend.clear_session()

def main():
    # Create folders
    if not os.path.exists(dir_models):
        os.mkdir(dir_models)
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)

    # Get the architecture name
    architecture = None
    if len(sys.argv) == 2:
        name = sys.argv[1]
        if name in architectures.keys():
            architecture = architectures[name]
    
    if architecture == None:
        quit()

    print(f'Preparing to train {name}.')

    # Delete existing results
    fname = f'{dir_results}{name}'
    if not os.path.exists(fname):
        os.mkdir(fname)
    fname = f'{dir_results}{name}/training.txt'
    if os.path.exists(fname):
        os.remove(fname)

    # Create the new results file.
    fname = f'{dir_results}{name}/training.txt'
    f = open(fname, 'a')
    f.write('epoch, samples, loss, train accuracy, test accuracy, time ms\n')
    f.close()

    # Load the data
    print("Loading group_index.json")
    file = open(f'{dir_braid}group_index.json')
    group_index = json.load(file)
    file.close()

    print("Loading training_x.npy")
    training_x = np.load(f'{dir_braid}training_x.npy')

    print("Loading training_y.npy")
    training_y = np.load(f'{dir_braid}training_y.npy')

    print("Loading testing_id.npy")
    testing_id = np.load(f'{dir_braid}testing_id.npy')

    print("Loading testing_x.npy")
    testing_x = np.load(f'{dir_braid}testing_x.npy')

    print("Loading testing_y.npy")
    testing_y = np.load(f'{dir_braid}testing_y.npy')

    process_model(name, architecture, group_index, training_x, training_y, testing_x, testing_y)


if __name__ == "__main__":
    main()
