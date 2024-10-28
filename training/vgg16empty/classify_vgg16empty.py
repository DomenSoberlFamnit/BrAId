import os
import numpy as np
import tensorflow as tf
import json
from PIL import Image

dir_braid = '/home/hicup/disk/braid/'
dir_model = f'{dir_braid}models/vgg16empty/'
dir_results = f'{dir_braid}results/vgg16empty/'
dir_photos = f'{dir_braid}results/vgg16empty/photos/'
dir_photos_hit = f'{dir_braid}results/vgg16empty/photos/hit/'
dir_photos_miss = f'{dir_braid}results/vgg16empty/photos/miss/'

def obscure_image(x):
    for row in range(224):
        for col in range (224):
            if row < 56 or col >= 112:
                x[row][col] = np.array([0, 0, 0])

def main():
    print("Classifying all instances using the saved VGG16 model.")

    # Create folders
    if not os.path.exists(dir_photos):
        os.mkdir(dir_photos)
    if not os.path.exists(dir_photos_hit):
        os.mkdir(dir_photos_hit)
    if not os.path.exists(dir_photos_miss):
        os.mkdir(dir_photos_miss)

    # Load the data
    print("Loading group_index.json")
    file = open(f'{dir_braid}group_index.json')
    group_index = json.load(file)
    file.close()

    group_index = dict(sorted(group_index.items(), key=lambda item: item[1], reverse=False))
    groups = list(group_index.keys())

    print("Loading data_id.npy")
    data_id = np.load(f'{dir_braid}data_id.npy')

    print("Loading data_x.npy")
    data_x = np.load(f'{dir_braid}data_x.npy')

    print("Loading data_y.npy")
    data_y = np.load(f'{dir_braid}data_y.npy')

    print("Loading the model.")
    model = tf.keras.models.load_model(f'{dir_model}vgg16empty.keras')
    print(model.summary())

    cnt, hit = 0, 0
    for (id, x, y) in zip(data_id, data_x, data_y):
        #obscure_image(x)
        prediction = model.predict(np.array([x]), verbose=0)[0]

        true_group = np.argmax(y)
        predicted_group = np.argmax(prediction)

        cnt += 1
        if predicted_group == true_group:
            hit += 1

            #if not os.path.exists(f'{dir_photos_hit}{groups[true_group]}/'):
            #    os.mkdir(f'{dir_photos_hit}{groups[true_group]}/')

            #img = tf.keras.preprocessing.image.array_to_img(x)
            #img.save(f'{dir_photos_hit}{groups[true_group]}/{id}_{groups[true_group]}_{groups[predicted_group]}.png')
        else:
            if not os.path.exists(f'{dir_photos_miss}{groups[true_group]}/'):
                os.mkdir(f'{dir_photos_miss}{groups[true_group]}/')

            img = tf.keras.preprocessing.image.array_to_img(x)
            img.save(f'{dir_photos_miss}{groups[true_group]}/{id}_{groups[true_group]}_{groups[predicted_group]}.png')
        
        if cnt % 1000 == 0:
            print(f'Classified instances {cnt}/{len(data_x)}; CA = {hit / cnt}')

if __name__ == "__main__":
    main()
