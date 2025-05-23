import os
import math
import requests
import hashlib
from progressbar import ProgressBar, Percentage, Bar
from ultralytics import YOLO
import tensorflow as tf

_models_url = 'https://osebje.famnit.upr.si/~domen.soberl/braid/'
_models_dirs = [
    'models/sentvid'
]
_models_files = [
    ('models/sentvid/VGG19.keras',   '632955d52056f80894ae003d9c0419ca9c805505bea1307f5e42fdca44fbc6ee'),
    ('models/sentvid/VGG19-1.keras', 'b765a140266b1ab7a3e4b34f297dbf95dc6a2a9fee02893c1fdb5beffccfa734'),
    ('models/sentvid/VGG19-2.keras', '533f56c8e33acd32b178dabb4af96e6276797408f0e753e6eb18f51d717f6b3d'),
    ('models/sentvid/VGG19-3.keras', '966a41260db325e6e4f5e0104ed2b50c753e6950a6c52edae68b8ccca59fdaf5'),
    ('models/sentvid/VGG19-4.keras', '554f3bb6dec74464440bc5f79ab94fc4759694bbeac35d0f86bb7043c3c7e546'),
    ('models/sentvid/VGG19-5.keras', 'cf92b2e759f77d97c31b78c5ad508828fac3b673efc937c0b4826e96cb918081'),
    ('models/sentvid/VGG19-6.keras', '690e6d0a133c167ede86caf25f2c4c94725aec732c1b0fbdbdf580bc4ba5d0c8'),
    ('models/sentvid/VGG19-7.keras', '851225d163790043da24488a6ebd4bfcfca44bb6b16626a3ce6afe5b79f6604a'),
    ('models/sentvid/VGG19-8.keras', '8b80d0e2197401d477a7c1b29a988892e282dfb8378539b86cd24cffb9220cb9'),
    ('models/sentvid/VGG19-9.keras', '30cb56202e621ef03fa51fe0ce2d1d872819faf87d2c0fefa2aabf262e129936'),
    ('models/sentvid/VGG19-10.keras', 'eb34b0ada4c853d1c585327a08cb4bfa16d9c2f4bcd935b4ba0add462fdf6d2d')
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

    with open(file, 'wb') as f:
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
        for dir in _models_dirs:
            os.makedirs(name=dir, exist_ok=True)

        braid_models = {}

        # Download the weights.
        screen_msg = False
        for (file, hash) in _models_files:
            file_ok = os.path.exists(file) and _compute_file_hash(file) == hash

            if not file_ok:
                if not screen_msg:
                    print('Downloading models:')
                    screen_msg = True

                _download_model(file)

        # Load the weights.
        for (file, _) in _models_files:
            braid_models[file] = tf.keras.models.load_model(file)

        # Store the loaded models.
        _models['braid'] = braid_models

    return _models
