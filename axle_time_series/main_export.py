import os
import numpy as np

import dataset
from models.tcn import TCN

dir_braid = '/mnt/workspace/braid/workdir/'
signal_length = 1300

force_training = True

#####################   Data preparation   ########################################

# Load or generate the vehicle index (a leftover from the camera project).
vehicle_info = dataset.load_vehicle_info(dir_braid, f'{dir_braid}camera/')
if vehicle_info is None:
    print('Cannot load vehicle info.')
    quit()

data = dataset.export_data(
    dir_braid=dir_braid,
    input_signal='11admp',
    normalized_signals=True,
    include_correct=True,
    include_fixed=False,
    signal_length=signal_length,
    vehicle_info=vehicle_info
)


