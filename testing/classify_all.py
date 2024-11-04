import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import shutil
import numpy as np
import tensorflow as tf
import json

dir_braid = '/home/hicup/disk/braid/'
dir_models = f'{dir_braid}models/'
dir_results = f'{dir_braid}results/'

architectures = [
    'VGG16',
    'VGG19',
    'DenseNet121',
    'MobileNetV3Small',
    'ResNet101V2'
]

def process_model(name, groups, data_id, data_x, data_y):
    dir_photos = f'{dir_results}{name}/photos_all/'
    dir_photos_hit = f'{dir_photos}hit/'
    dir_photos_miss = f'{dir_photos}miss/'

    # Remove existing photos
    if os.path.exists(dir_photos):
        shutil.rmtree(dir_photos)

    # Create empty directories
    os.mkdir(dir_photos)
    os.mkdir(dir_photos_hit)
    os.mkdir(dir_photos_miss)

    # Load the model
    print(f'Loading the model {name}.')
    model = tf.keras.models.load_model(f'{dir_models}{name}.keras')

    cnt, hit = 0, 0
    for (id, x, y) in zip(data_id, data_x, data_y):
        #obscure_image(x)
        prediction = model.predict(np.array([x]), verbose=0)[0]

        true_group = np.argmax(y)
        predicted_group = np.argmax(prediction)

        cnt += 1
        if predicted_group == true_group:
            hit += 1

            if not os.path.exists(f'{dir_photos_hit}{groups[true_group]}/'):
                os.mkdir(f'{dir_photos_hit}{groups[true_group]}/')

            img = tf.keras.preprocessing.image.array_to_img(x)
            img.save(f'{dir_photos_hit}{groups[true_group]}/{id}_{groups[true_group]}_{groups[predicted_group]}.png')
        else:
            if not os.path.exists(f'{dir_photos_miss}{groups[true_group]}/'):
                os.mkdir(f'{dir_photos_miss}{groups[true_group]}/')

            img = tf.keras.preprocessing.image.array_to_img(x)
            img.save(f'{dir_photos_miss}{groups[true_group]}/{id}_{groups[true_group]}_{groups[predicted_group]}.png')
        
        if cnt % 1000 == 0:
            print(f'Classified instances {cnt}/{len(data_x)}; CA = {hit / cnt}')

        tf.keras.backend.clear_session()

def main():
    print("Preparing to test the models on testing data.")

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

    for name in architectures:
        process_model(name, groups, data_id, data_x, data_y)


if __name__ == "__main__":
    main()