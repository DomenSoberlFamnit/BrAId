import os
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
from classifier import axle_groups_from_image

dir_braid = '/home/hicup/disk/braid/'
dir_models = f'{dir_braid}models/'
dir_photos = '/home/hicup/disk/validation/'

architectures = [
    'VGG16',
    'VGG19',
    'DenseNet121',
    'MobileNetV3Small',
    'ResNet101V2'
]

yolo = YOLO("yolov8x.pt")

models = []
for arch in architectures:
    models.append((arch, tf.keras.models.load_model(f'{dir_models}{arch}.keras')))

photos = []

for root, dirs, files in os.walk(dir_photos, topdown=False):
    for name in files:
        filename = os.path.join(root, name)
        photos.append((f"{root.split('/')[-1]}/{name}", Image.open(filename)))

f = open("output.txt", "w")

f.write('File')
for arch in architectures:
    f.write(f'\t{arch}\tProbability')
f.write('\n')

for (name, image) in photos:
    print(name)
    f.write(f'{name}')
    for (arch, model) in models:
        res = axle_groups_from_image(image, 'sentvid', yolo, model)
        if res['success']:
            (groups, prob) = res['axle_groups'][0]
            f.write(f'\t{groups}\t{prob}')
        else:
            f.write(f'\t\t')
    f.write('\n')

f.close()
