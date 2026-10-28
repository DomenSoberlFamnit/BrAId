import os
import math
import requests
import hashlib
from progressbar import ProgressBar, Percentage, Bar
from ultralytics import YOLO
import tensorflow as tf

_models_url = 'https://github.com/DomenSoberlFamnit/BrAId/releases/download/v1.0/'
_models_files = [
    ('models/sentvid-VGG19.keras', 'sentvid-VGG19.keras', '632955d52056f80894ae003d9c0419ca9c805505bea1307f5e42fdca44fbc6ee')
]

_group_index = {
    'sentvid': ["113", "1211", "122", "11", "22", "111", "112", "1112", "12", "1111", "123", "1212", "1222"]
}

_models = None


def _create_progress_bar(task_name, max_value):
    pbar = ProgressBar(
        widgets=[
            task_name,
            Percentage(), ' ',
            Bar(marker='#', left='[', right=']')
        ],
        maxval=max_value
    )
    pbar.start()

    return pbar


def _compute_file_hash(file_path):
    hash_func = hashlib.new('sha256')

    with open(file_path, 'rb') as file:
        while chunk := file.read(1024*1024):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def _download_model(file):
    response = requests.get(url=f'{_models_url}{file}', stream=True)
    file_size = int(response.headers['Content-Length'])
    chunk_size = 1024*1024
    chunk_total = math.ceil(file_size / chunk_size)

    pbar = _create_progress_bar(f'{file}: ', chunk_total)

    with open(f'models/{file}', 'wb') as f:
        chunk_count = 0
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                chunk_count += 1
                pbar.update(chunk_count)

    pbar.finish()


def _init_models():
    global _models

    if _models is None:
        _models = {
            'yolo': None,
            'braid': None
        }

    # Check and download YOLOv8 weights.
    if _models['yolo'] is None:
        _models['yolo'] = YOLO("yolov8x.pt")

    # Check and download BrAId weights.
    if _models['braid'] is None:
        os.makedirs(name='models', exist_ok=True)

        braid_models = {}

        # Download the weights.
        screen_msg = False
        for (file, remote, hash) in _models_files:
            file_ok = os.path.exists(file) and _compute_file_hash(file) == hash

            if not file_ok:
                if not screen_msg:
                    print('Downloading models:')
                    screen_msg = True

                _download_model(remote)

        # Load the weights.
        for (file, _, _) in _models_files:
            braid_models[file] = tf.keras.models.load_model(file)

        # Store the loaded models.
        _models['braid'] = braid_models

    return _models
