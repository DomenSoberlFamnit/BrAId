import os
from PIL import Image
import braid

dir_braid = '/home/hicup/disk/braid/'
dir_photos = f'{dir_braid}problematic_photos/'

cnt = 0

for _, _, files in os.walk(dir_photos):
    for file in files:
        if file.endswith('.png'):
            parts = file.split('.')[0].split('-')
            truth = parts[1]

            filepath = f'{dir_photos}{file}'
            print(filepath)

            img = Image.open(filepath)

            braid_results = braid.axle_groups_from_image(image=img, site='sentvid')

            if 'axle_groups' in braid_results:
                (axle_groups, probability) = braid_results['axle_groups'][0]
                print(f"{axle_groups} {probability}")

                if axle_groups == truth:
                    cnt += 1
            else:
                print("yolo error")

print("Correct:", cnt)