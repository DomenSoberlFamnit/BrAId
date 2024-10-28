import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import json

dir_braid = '/home/hicup/disk/braid/'
dir_model = f'{dir_braid}models/vgg19/'
dir_results = f'{dir_braid}results/vgg19/'

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

def train(model, train_x, train_y, gpu_capacity=5000):
    loss, ca = 0, 0
    for (idx_from, idx_to, ratio) in slot_indices(gpu_capacity, len(train_x)):
        history = model.fit(x=train_x[idx_from:idx_to], y=train_y[idx_from:idx_to], batch_size=32, epochs=1, validation_split=0, shuffle=False)
        loss += ratio * history.history['loss'][0]
        ca += ratio * history.history['accuracy'][0]
    return loss, ca

def test(model, test_x, test_y, gpu_capacity=5000):
    correct = 0
    for (idx_from, idx_to, _) in slot_indices(gpu_capacity, len(test_x)):
        predictions = model.predict(test_x[idx_from:idx_to])
        for (prediction, true_y) in zip(predictions, test_y[idx_from:idx_to]):
            if np.argmax(prediction) == np.argmax(true_y):
                correct += 1
    
    return correct/len(test_x)

def build_model(cls_count):
    # Load the VGG19 model pre-trained on ImageNet without the top classification layer
    model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model
    # model.trainable = False

    # Add new classification layers on top of the base model
    x = model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(cls_count, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Return the model
    return model

def main():
    print("Preparing to train the VGG19 model.")

    # Create folders
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)

    # Delete existing results
    if os.path.exists(f'{dir_results}vgg19_ca.txt'):
        os.remove(f'{dir_results}vgg19_ca.txt')

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

    # Build the network architecture
    model = build_model(len(group_index))

    for epoch in range(25):
        print(f'Epoch {epoch} training.')
        train_loss, train_accuracy = train(model, training_x, training_y)
        print(f'Epoch {epoch} testing on training set.')
        train_accuracy_manual = test(model, training_x, training_y)
        print(f'Epoch {epoch} testing on testing set.')
        test_accuracy = test(model, testing_x, testing_y)

        f = open(f'{dir_results}vgg19_ca.txt', 'a')
        f.write(f'{epoch + 1}, {train_loss}, {train_accuracy}, {train_accuracy_manual}, {test_accuracy}\n')
        f.close()

    model.save(f'{dir_model}vgg19.keras')

if __name__ == "__main__":
    main()
