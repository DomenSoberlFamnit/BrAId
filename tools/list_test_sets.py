import numpy as np

dir_braid = '/home/hicup/disk/braid/'

print("Loading testing_id.npy")
testing_id = np.load(f'{dir_braid}data/testing_id.npy')

print("Loading testing_y.npy")
testing_y = np.load(f'{dir_braid}data/testing_y.npy')


