import os
import json

dir_braid = '/home/hicup/disk/braid-old/'
architecture = 'VGG19'

def collect_from_dir(dir):
    instances = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            parts = name.split(".")[0].split("_")
            if name.endswith('.png') and len(parts) == 4:
                id = parts[0]
                label = parts[1]
                ai = parts[2]
                instances.append({'ID': id, 'class': label})
    return instances

data = {}
cnt = 0
for experiment in range(1, 11):
    instances = collect_from_dir(f'{dir_braid}results{experiment}/{architecture}/')
    cnt += len(instances)
    data[experiment] = instances

print(f'Instance count {cnt}.')

with open("testing_ids.json", "w") as f:
    json.dump(data, f)
