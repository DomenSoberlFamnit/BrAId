import os
import json
from PIL import Image

def find_vehicle(rv, id):
    for data in rv:
        if str(data['photo_id']) == id:
            return data
    return None

def get_box(segments, color):
    for seg in segments:
        box = seg['box']
        if box['color'] == color:
            return (box['x'], box['y'], box['width'], box['height'])
    return None

def img_resize_224(img):
    ratio = img.width / img.height

    out = Image.new(mode="RGB", size=(224, 224), color="black")

    if ratio == 1:
        img1 = img.resize((224, 224))
        out.paste(img1)
    elif ratio > 1:
        img1 = img.resize((224, round(224/ratio)))
        out.paste(img1, (0, round((224 - img1.height)/2)))
    else:
        img1 = img.resize((round(224*ratio), 224))
        out.paste(img1, (round((224 - img1.width)/2), 0))

    return out

def get_photo(src_photos_dir, rv_record, id, seg):
    subdir = str(int(int(id)/1000)) + '/'
    try:
        img = Image.open(src_photos_dir + subdir + str(id) + '.png')
    except:
        return None

    (x, y, w, h) = get_box(rv_record['segments'], seg)
    cropped = img.crop((x, y, x+w, y+h))

    return img_resize_224(cropped)

def run(dir_braid):
    src_photos_dir = dir_braid + 'photos/'
    dst_photos_dir = dir_braid + 'cropped_photos/'

    if not os.path.exists(dst_photos_dir):
        os.mkdir(dst_photos_dir)

    f = open(f'{dir_braid}recognized_vehicles.json')
    rv = json.load(f)
    f.close()

    f = open(f'{dir_braid}valid_photos.json')
    valid_photos = json.load(f)
    f.close()

    i = 0
    n = len(valid_photos)

    for record in valid_photos:
        photo_id = record['photo_id']
        segment = record['segment']
        groups = record['class']

        i += 1
        rv_record = find_vehicle(rv, photo_id)

        img = get_photo(src_photos_dir, rv_record, photo_id, segment)
        if img != None:
            if not os.path.exists(f'{dst_photos_dir}{groups}/'):
                os.mkdir(f'{dst_photos_dir}{groups}/')
            img.save(f'{dst_photos_dir}{groups}/{photo_id}.png')
            
            if i % 1000 == 0:
                print(f'Cropped {i}/{n} photos.')