from ultralytics import YOLO
from PIL import Image, ImageDraw
import math
import json
import os

def is_duplicate(segment1, segment2):
    box1 = segment1['box']
    box2 = segment2['box']
    
    x1 = box1['x']
    y1 = box1['y']
    width1 = box1['width']
    height1 = box1['height']

    x2 = box2['x']
    y2 = box2['y']
    width2 = box2['width']
    height2 = box2['height']

    tolerance = 5

    if (abs(x2 - x1) > tolerance):
        return False
    if (abs(y2 - y1) > tolerance):
        return False
    if (abs((x2 + width2) - (x1 + width1)) > tolerance):
        return False
    if (abs((y2 + height2) - (y1 + height1)) > tolerance):
        return False

    return True

def segment_contained(segment, segments):
    for existing in segments:
        if is_duplicate(segment, existing):
            return True
    return False

def remove_duplicates(segments):
    new_segments = []

    change = False
    for segment in segments:
        if not segment_contained(segment, new_segments):
            new_segments.append(segment)
        else:
            change = True

    return change, new_segments

def run(dir_braid):
    src_photos_dir = dir_braid + 'photos/'
    dst_photos_dir = dir_braid + 'yolo_photos/'

    if not os.path.exists(dst_photos_dir):
        os.mkdir(dst_photos_dir)

    valid_timestamps = []
    with open('valid_timestamps.txt', 'r') as file:
        for timestamp in file.read().strip().split(','):
            valid_timestamps.append(float(timestamp))

    file = open(f'{dir_braid}vehicle_index.json')
    vehicle_index = json.load(file)
    file.close()

    model = YOLO("yolov8x.pt")
    data = []

    photo_cnt = 0
    for vehicle in vehicle_index:
        photo_fn = src_photos_dir + vehicle['file']

        print(photo_fn + ' ... ', end='')

        if not vehicle['ts_photo'] in valid_timestamps:
            print('not valid')
            continue

        try:
            img = Image.open(photo_fn)
        except FileNotFoundError:
            print("file error")
            continue

        results = model.predict(source=img, verbose=False, save=False)
        result = results[0]

        color_codes = ['r', 'g', 'b', 'c', 'y', 'm', 'w', 'b', 't']
        color_names = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'white', 'black', 'teal']
        count_recognitions = 0
        vehicle_boxes = []

        for (box, conf, cls) in zip(result.boxes, result.boxes.conf, result.boxes.cls):
            cls = result.names[int(cls)]
            probability = round(float(conf)*10000)/100
            x = int(box.xywh[0][0].int())
            y = int(box.xywh[0][1].int())
            w = int(box.xywh[0][2].int())
            h = int(box.xywh[0][3].int())

            x = round(x - w/2)
            y = round(y - h/2)

            if cls == "truck" or cls=="bus":
                vehicle_boxes.append({'type':cls, 'probability':probability, 'box':{'x': x, 'y': y, 'width':w, 'height':h, 'color':color_codes[count_recognitions]}})
                print(cls, end=' ')
                count_recognitions += 1

        if len(vehicle_boxes) > 0:
            sorted_segments = sorted(vehicle_boxes, key=lambda x: x['probability'], reverse=True)
            change, segments = remove_duplicates(sorted_segments)

            if change:
                print(' [duplicates removed] ', end='')

            dir = dst_photos_dir + str(math.floor(int(vehicle['id']) / 1000)) + '/'
            if not os.path.exists(dir):
                os.makedirs(dir)

            draw = ImageDraw.Draw(img)
            for i, segment in enumerate(segments):
                box = segment['box']
                x = box['x']
                y = box['y']
                w = box['width']
                h = box['height']
                draw.rectangle([(x, y), (x + w, y + h)], outline=color_names[i], width=2)
                box['color'] = color_codes[i]
                segment['box'] = box
            
            img.save(dir + str(vehicle['id']) + '.png')

            data.append({'photo_id':vehicle['id'], 'photo_timestamp':vehicle['ts_photo'], 'vehicle_timestamp':vehicle['ts_vehicle'], 'axle_count':vehicle['axles'], 'axle_groups':vehicle['groups'], 'gvw':vehicle['gvw'],'segments':segments})

            print("done")
            photo_cnt += 1

        else:
            print("no vehicles")

    with open(f'{dir_braid}recognized_vehicles.json', 'w') as file:
        json.dump(data, file)
