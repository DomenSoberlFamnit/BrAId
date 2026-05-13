from swm import factory
from datetime import datetime
import os
import json

def sortkey(item):
    return item['timestamp']

def run(dir_siwim, dir_braid):
    index = []

    for root, dirs, files in os.walk(dir_siwim + "sites/AC_Sentvid_2012_2/live/", topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            filenameshort = filename.replace(dir_siwim + "sites/AC_Sentvid_2012_2/live/", "")
            try:
                data = factory.read_file(filename)
            except:
                print("Error:", filename)
                continue

            # print(data.vts.timestamp(), filenameshort)
            index.append({'timestamp': data.vts.timestamp(), 'filename': filenameshort})

            if len(index) % 1000 == 0:
                print(f'Processed {len(index)} photos.')

    print("Sorting photo index.")
    index.sort(key=sortkey)

    print("Saving photo_index.json.")
    with open(dir_braid + "photo_index.json", "w") as fp:
        json.dump(index, fp)  # encode dict into JSON
