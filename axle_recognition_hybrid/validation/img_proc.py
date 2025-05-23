from PIL import Image


def _is_duplicate(segment1, segment2):
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


def _segment_contained(segment, segments):
    for existing in segments:
        if _is_duplicate(segment, existing):
            return True
    return False


def _remove_duplicates(segments):
    new_segments = []

    change = False
    for segment in segments:
        if not _segment_contained(segment, new_segments):
            new_segments.append(segment)
        else:
            change = True

    return change, new_segments


def _img_resize_224(img):
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


def _yolo(yolo_model, img):
    results = yolo_model.predict(source=img, verbose=False, save=False)
    result = results[0]

    count_recognitions = 0
    vehicle_boxes = []

    for (box, conf, cls) in zip(result.boxes, result.boxes.conf, result.boxes.cls):
        cls = result.names[int(cls)]
        probability = round(float(conf)*10000)/100
        x = int(box.xywh[0][0].int())
        y = int(box.xywh[0][1].int())
        w = int(box.xywh[0][2].int())
        h = int(box.xywh[0][3].int())
        area = w*h

        x = round(x - w/2)
        y = round(y - h/2)

        if cls == "truck" or cls == "bus":
            vehicle_boxes.append({'type':cls, 'area':area, 'probability':probability, 'box':{'x': x, 'y': y, 'width':w, 'height':h}})
            count_recognitions += 1

    if len(vehicle_boxes) > 0:
        sorted_segments = sorted(vehicle_boxes, key=lambda x: x['area'], reverse=True)
        _, segments = _remove_duplicates(sorted_segments)
        segment = segments[0]

        vehicle = segment['type']
        probability = segment['probability']

        box = segment['box']
        x = box['x']
        y = box['y']
        w = box['width']
        h = box['height']

        return (vehicle, probability, x, y, w, h)
    else:
        return None
