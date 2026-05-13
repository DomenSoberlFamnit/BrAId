import numpy as np
import tensorflow as tf
from PIL import Image
from img_proc import _yolo, _img_resize_224
import time

tf.config.set_visible_devices([], 'GPU')

_group_index = {
    'sentvid': ["113", "1211", "122", "11", "22", "111", "112", "1112", "12", "1111", "123", "1212", "1222"]
}

def axle_groups_from_image(
        image: Image,
        site: str,
        yolo: object,
        model: object) -> dict:

    if model is None:
        return {'success': False, 'msg': 'Error: Model not loaded.'}

    if site not in _group_index:
        return {'success': False, 'msg': 'Error: Unknown site.'}

    groups = _group_index[site]

    yolo_return = _yolo(yolo, image)
    if yolo_return is None:
        return {'success': False, 'msg': 'Vehicle not recognised.'}

    (vehicle, probability, x, y, w, h) = yolo_return

    cropped = image.crop((x, y, x+w, y+h))
    resized = _img_resize_224(cropped)
    instance = tf.keras.preprocessing.image.img_to_array(resized)

    ret = {
        'success': True,
        'msg': '',
        'vehicle_type': vehicle,
        'type_probability': probability,
        'segment': {'x': x, 'y': y, 'w': w, 'h': h},
        'axle_groups': []
    }

    t = time.time()
    prediction = model.predict(np.array([instance]), verbose=0)[0]
    print("time", time.time() - t)
    predicted_group = np.argmax(prediction)
    prediction_probability = float(prediction[predicted_group])

    ret['axle_groups'].append((groups[predicted_group], prediction_probability))

    return ret
