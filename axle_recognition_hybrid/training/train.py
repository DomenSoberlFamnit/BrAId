import random
import json
import tensorflow as tf
import tf_models
from PIL import Image, ImageDraw
import numpy as np
from timeit import default_timer as timer
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import dark_attention

dir_braid = '/home/hicup/disk/braid/'
dir_models = f'{dir_braid}models/'
dir_results = f'{dir_braid}results/'

epochs = 20
sample_count = 0


def update_dirs(experiment):
    global dir_models, dir_results
    dir_models = f'{dir_braid}models{experiment}/'
    dir_results = f'{dir_braid}results{experiment}/'


def alter_image(img):
    width, height = img.size
    new_width = round(random.uniform(0.9, 1.0) * width)
    new_height = round(random.uniform(0.9, 1.0) * height)
    img = img.resize((new_width, new_height))

    space_x = width - new_width
    space_y = height - new_height
    offset_x = round(random.uniform(0.0, space_x))
    offset_y = round(random.uniform(0.0, space_y))

    new_img = Image.new(mode="RGB", size=(width, height), color="black")
    new_img.paste(img, (offset_x, offset_y))
    return new_img


def alter_batch(batch):
    new_batch = []
    for sample in batch:
        img = tf.keras.preprocessing.image.array_to_img(sample)
        img = alter_image(img)
        new_sample = tf.keras.preprocessing.image.img_to_array(img)
        new_batch.append(new_sample)
    return np.array(new_batch)


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


def test(model, test_x, test_y, gpu_capacity=10000):
    correct = 0

    for (idx_from, idx_to, _) in slot_indices(gpu_capacity, len(test_x)):
        predictions = model.predict(test_x[idx_from:idx_to])
        for (prediction, true_y) in zip(predictions, test_y[idx_from:idx_to]):
            if np.argmax(prediction) == np.argmax(true_y):
                correct += 1

    return correct / len(test_x)


def train(model, name, epoch, train_x, train_y, testing_x, testing_y, gpu_capacity=10000):
    global sample_count

    for (idx_from, idx_to, _) in slot_indices(gpu_capacity, len(train_x)):
        batch_x = alter_batch(train_x[idx_from:idx_to])

        time_start = timer()
        history = model.fit(x=batch_x, y=train_y[idx_from:idx_to],
                            batch_size=32, epochs=1, validation_split=0, shuffle=False)
        time_end = timer()

        time_ms = 1000 * (time_end - time_start) / (idx_to - idx_from)
        sample_count += idx_to - idx_from

        loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]

        if len(testing_y) > 0:
            test_accuracy = test(model, testing_x, testing_y)
        else:
            test_accuracy = 0

        fname = f'{dir_results}{name}/training.txt'
        f = open(fname, 'a')
        f.write(
            f'{epoch + 1}, {sample_count}, {loss}, {train_accuracy}, {test_accuracy}, {time_ms}\n')
        f.close()


def build_model(name, group_index):
    model = tf_models.build_model(name, len(group_index))

    if model is not None:
        model.summary()
    
    return model


def process_model(model, name, training_x, training_y, testing_x, testing_y):
    global sample_count

    sample_count = 0
    for epoch in range(epochs):
        print(f'Training epoch {epoch + 1}/{epochs}.')
        train(model, name, epoch, training_x, training_y, testing_x, testing_y)

    print(f'Saving the model {name}.')
    #model.save(f'{dir_models}{name}.keras')
    model.save_weights(f'{dir_models}{name}.weights.h5')

    tf.keras.backend.clear_session()


def main():
    # Get the architecture name
    experiment = None
    if len(sys.argv) >= 2:
        name = sys.argv[1]
        if len(sys.argv) > 2:
            experiment = sys.argv[2]

    if experiment is not None:
        update_dirs(experiment)

    # Load the group index
    print("Loading group_index.json")
    file = open(f'{dir_braid}group_index.json')
    group_index = json.load(file)
    file.close()

    # Build the model
    print(f'Building the model {name}.')
    model = build_model(name, group_index)
    if model is None:
        print('Unknow model type.')
        quit()

    print(f'Preparing to train {name}.')

    # Create folders
    if not os.path.exists(dir_models):
        os.mkdir(dir_models)
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)
    
    # Delete existing results
    fname = f'{dir_results}{name}'
    if not os.path.exists(fname):
        os.mkdir(fname)
    fname = f'{dir_results}{name}/training.txt'
    if os.path.exists(fname):
        os.remove(fname)

    # Create the new results file.
    fname = f'{dir_results}{name}/training.txt'
    print(f'Creating file {fname}.')
    f = open(fname, 'a')
    f.write('epoch, samples, loss, train accuracy, test accuracy, time ms\n')
    f.close()

    print("Loading training_x.npy")
    training_x = np.load(f'{dir_braid}data/training_x.npy')

    print("Loading training_y.npy")
    training_y = np.load(f'{dir_braid}data/training_y.npy')

    print("Loading testing_id.npy")
    testing_id = np.load(f'{dir_braid}data/testing_id.npy')

    print("Loading testing_x.npy")
    testing_x = np.load(f'{dir_braid}data/testing_x.npy')

    print("Loading testing_y.npy")
    testing_y = np.load(f'{dir_braid}data/testing_y.npy')

    process_model(model, name, training_x,
                  training_y, testing_x, testing_y)


if __name__ == "__main__":
    main()
