import h5py
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf

dir_braid = '/home/hicup/disk/braid/'
dir_photos = f'{dir_braid}photos/'
dir_models = f'{dir_braid}models/'

model_name = 'DenseNet121'

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

def get_vehicle_photo(id, yolo):
    img = Image.open(f'{dir_photos}{id[0:len(id)-3] if len(id) > 3 else "0"}/{id}.png')
    yolo_result = yolo.predict(source=img, verbose=False, save=False)[0]

    vehicle_boxes = []
    for (box, conf, cls) in zip(yolo_result.boxes, yolo_result.boxes.conf, yolo_result.boxes.cls):
        cls = yolo_result.names[int(cls)]
        probability = round(float(conf)*10000)/100
        x = int(box.xywh[0][0].int())
        y = int(box.xywh[0][1].int())
        w = int(box.xywh[0][2].int())
        h = int(box.xywh[0][3].int())

        x = round(x - w/2)
        y = round(y - h/2)

        if cls == "truck" or cls=="bus":
            vehicle_boxes.append({'type':cls, 'probability':probability, 'size': (w * h), 'box':{'x': x, 'y': y, 'width':w, 'height':h}})

    if len(vehicle_boxes) > 0:
        sorted_segments = sorted(vehicle_boxes, key=lambda x: x['size'], reverse=True)
        _, segments = remove_duplicates(sorted_segments)
        segment = segments[0]
        box = segment['box']
        x, y, w, h = box['x'], box['y'], box['width'], box['height']

        #img.save("img.png")
        #cropped = img.crop((x, y, x+w, y+h))
        #cropped.save("cropped.png")
        #inst = img_resize_224(cropped)
        #inst.save("instance.png")

        return img_resize_224(img.crop((x, y, x+w, y+h)))
    else:
        return None        

def add_raised_axles(groups, raised_axles):
    raised_groups = list(groups)
    for axle in raised_axles.split(','):
        idx = int(axle) - 1
        raised_groups[idx] = str(int(raised_groups[idx]) + 1)
    return ''.join(raised_groups)

def main():
    with open(f'{dir_braid}group_index.json', 'r') as file:
        group_index = json.load(file)
    group_index = list(group_index.keys())

    yolo = YOLO("yolov8x.pt")
    model = tf.keras.models.load_model(f'{dir_models}{model_name}.keras')

    cnt_raised_axles = 0

    cnt_siwim_incorrect = 0
    cnt_hybrid_incorrect = 0
    cnt_siwim_correct_excluded = 0
    cnt_siwim_incorrect_excluded = 0
    
    cnt_camera_used = 0
    cnt_camera_failed = 0

    cnt_agree = 0
    cnt_disagree = 0
    cnt_raised_axles_agree = 0
    cnt_raised_axles_disagree = 0
    cnt_all = 0

    with h5py.File('../metadata/metadata.hdf5', 'r') as file:
        for siwim_groups in file.keys():
            data = file[siwim_groups]
            for id in data:
                prop = json.loads(file[f'{siwim_groups}/{id}'].asstr()[()])
                
                # We only use instances that were checked by an expert.
                if prop['seen_by'] == None:
                    continue

                # Did the expert determine raised axles?
                raised_axles = 'raised_axles' in prop and len(prop['raised_axles'].strip()) > 0
                if raised_axles:
                    cnt_raised_axles += 1

                # Ground truth
                true_groups = prop['axle_groups'] if 'axle_groups' in prop else siwim_groups

                siwim_correct = True
                if 'errors' in prop and 'fixed' in prop['errors']:
                    siwim_correct = False # The signals were fixed manually.
                elif true_groups != siwim_groups:
                    siwim_correct = False
                    # But check, if the groups changed because of the raised axles.
                    if raised_axles:
                        if add_raised_axles(siwim_groups, prop['raised_axles']) == true_groups:
                            siwim_correct = True

                if not siwim_correct:
                    cnt_siwim_incorrect += 1

                # Here, the SiWIM system outputs an axle group. Some classes may be incorrect.
                # We add the camera system for and additional check.

                # Get the photo.
                img = get_vehicle_photo(id, yolo)
                if img != None:
                    cnt_camera_used += 1

                    # Classify the image
                    x = tf.keras.preprocessing.image.img_to_array(img)
                    y = model.predict(np.array([x]), verbose=0)[0]
                    camera_groups = group_index[np.argmax(y)]

                    if siwim_groups == camera_groups:
                        # The instance is not excluded.
                        cnt_agree += 1

                        if raised_axles:
                            cnt_raised_axles_agree += 1

                        # If incorrect and not excluded by the camera.
                        if not siwim_correct:
                            cnt_hybrid_incorrect += 1
                    
                    else:
                        # The instance in excluded.
                        cnt_disagree += 1

                        if raised_axles:
                            cnt_raised_axles_disagree += 1

                        if siwim_correct:
                            cnt_siwim_correct_excluded += 1
                        else:
                            cnt_siwim_incorrect_excluded += 1

                else:
                    # Camera system not used.
                    cnt_camera_failed += 1

                    # If incorrect and not excluded by the camera.
                    if not siwim_correct:
                        cnt_hybrid_incorrect += 1
                
                # Count the instances that we used.
                cnt_all += 1
                if cnt_all % 100 == 0:
                    print(f'Processed {cnt_all}; agree / disagree: {cnt_agree} / {cnt_disagree} ({100 * cnt_agree / (cnt_agree + cnt_disagree):.2f}%)')

    print(f'All: {cnt_all}')
    print(f'SiWIM incorrect: {cnt_siwim_incorrect} ({100 * cnt_siwim_incorrect / cnt_all}%)')
    print(f'Hybrid incorrect: {cnt_hybrid_incorrect} ({100 * cnt_hybrid_incorrect / cnt_all}%)')
    print(f'Camera used: {cnt_camera_used} ({100 * cnt_camera_used / cnt_all}%)')
    print(f'Agree / disagree: {cnt_agree} / {cnt_disagree} ({100 * cnt_agree / (cnt_agree + cnt_disagree)}%)')
    print(f'Raised axles / agree / disagree: {cnt_raised_axles} / {cnt_raised_axles_agree} / {cnt_raised_axles_disagree}')
    print(f'Excluded correct / incorrect: {cnt_siwim_correct_excluded} / {cnt_siwim_incorrect_excluded}')

if __name__ == "__main__":
    main()