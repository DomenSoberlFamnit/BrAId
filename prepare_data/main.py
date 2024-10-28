import os
import photo_index
import vehicle_index
import recognized_vehicles
import crop_photos
import create_instances
import split_instances
import check_instances

dir_siwim = '/home/hicup/braid/siwim/'
dir_braid = '/home/hicup/disk/braid/'

force_instance_split = True  # Randomly split the instances even if the datasets already exist.
num_training_samples = 5000  # The size of the class. If less available, oversampling through replication is used.
min_testing_samples = 26     # The minimum number of instances kept for testing in each class.
max_testing_samples = 20000  # The maximum number of instances kept for testing in each class.

# Create the index of sorted photos: photo_index.json
if os.path.exists(f'{dir_braid}photo_index.json'):
    print("Found photo_index.json.")
else:
    print("Creating photo_index.json.")
    photo_index.run(dir_siwim, dir_braid)

# Create the index of vehicles: vehicle_index.json
# This will also extract all the photos.
if os.path.exists(f'{dir_braid}vehicle_index.json'):
    print("Found vehicle_index.json.")
else:
    print("Creating vehicle_index.json.")
    vehicle_index.run(dir_siwim, dir_braid)

# Create the index of recognized vehicles: recognized_vehicles.json
# It uses YOLO to draw segments. The yolo_photos folder is created.
# ZAG provides the valid_timestamps data, where only the vehicles
# from lane 1 of a certain time area are contained.
if os.path.exists(f'{dir_braid}recognized_vehicles.json'):
    print("Found recognized_vehicles.json.")
else:
    if not os.path.exists(f'valid_timestamps.txt'):
        print("File valid_timestamps.txt not found!")
        quit()

    print("Creating recognized_vehicles.json.")
    recognized_vehicles.run(dir_braid)

# Crop the photos created by YOLO. Only the photos that will be used
# for machine learning are cropped. The criteria which photos to use
# is hardcoded in the crop_photos script and based on the metadata
# metadata.hdf5, which is provided by ZAG.
#
# NOTE: At this point it is decided which photos will be used
#       for machine learning.
#
if os.path.exists(f'{dir_braid}cropped_photos'):
    print("Found the cropped_photos folder.")
else:
    if not os.path.exists(f'metadata.hdf5'):
        print("File metadata.hdf5 not found!")
        quit()

    print("Cropping photos.")
    crop_photos.run(dir_braid)

# Create training instances from cropped photos.
if (
    os.path.exists(f'{dir_braid}data_id.npy') and
    os.path.exists(f'{dir_braid}data_x.npy') and
    os.path.exists(f'{dir_braid}data_y.npy')
):
    print("Found data_id.npy, data_x.npy, data_y.npy")
else:
    print("Creating training instances.")
    create_instances.run(dir_braid)

# Prepare the training and the testing sets
if (
    not force_instance_split and
    os.path.exists(f'{dir_braid}training_id.npy') and
    os.path.exists(f'{dir_braid}training_x.npy') and
    os.path.exists(f'{dir_braid}training_y.npy') and
    os.path.exists(f'{dir_braid}testing_id.npy') and
    os.path.exists(f'{dir_braid}testing_x.npy') and
    os.path.exists(f'{dir_braid}testing_y.npy')
):
    print("Found training_id.npy, training_x.npy, training_y.npy, testing_id.npy, testing_x.npy, testing_y.npy")
else:
    print("Splitting the instances into training and testing sets.")
    split_instances.run(dir_braid, num_training_samples, min_testing_samples)

# Check the split instances.
print('Checking training and testing instances.')
check_instances.run(dir_braid)

# Create the directories to store models and results.
if not os.path.exists(f'{dir_braid}models/'):
    os.mkdir(f'{dir_braid}models/')

if not os.path.exists(f'{dir_braid}results/'):
    os.mkdir(f'{dir_braid}results/')
