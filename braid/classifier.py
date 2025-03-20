import numpy as np
import tensorflow as tf
from PIL import Image
from braid.init import _init_models, _group_index
from braid.img_proc import _yolo, _img_resize_224


def axle_groups_from_image(
        image: Image,
        site: str,
        architecture: str = 'VGG19',
        variants: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) -> dict:

    models = _init_models()
    if models is None:
        return {'success': False, 'msg': 'Error: Models not loaded.'}

    if site not in _group_index:
        return {'success': False, 'msg': 'Error: Unknown site.'}

    groups = _group_index[site]

    yolo_return = _yolo(models['yolo'], image)
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

    for variant in variants:
        model_file = f'models/{site}/{architecture}-{variant}.keras'

        if model_file not in models['braid']:
            return {'success': False, 'msg': 'Error: Unknown model.'}

        model = models['braid'][model_file]

        prediction = model.predict(np.array([instance]), verbose=0)[0]
        predicted_group = np.argmax(prediction)
        prediction_probability = float(prediction[predicted_group])

        ret['axle_groups'].append((variant, groups[predicted_group], prediction_probability))

    return ret
