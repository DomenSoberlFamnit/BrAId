import os
import shutil
import random

dir_braid = '/home/hicup/disk/braid/'

def index_contains(index, id):
    for (idx_id, _, _) in index:
        if idx_id == id:
            return True
    return False

def update_case_index(results_dir, case_index):
    for (root, _, files) in os.walk(results_dir):
        for file in files:
            if file.endswith('.png'):
                parts = file.split('.')[0].split('_')
                if len(parts) == 4:
                    filepath = f'{root}/{file}'
                    id = parts[0]
                    truth = parts[1]
                    if not index_contains(case_index, id):
                        case_index.append((id, filepath, file))

def add_missing_cases(results_dir, case_index, hit_index):
    while len(case_index) < 364:
        id = None
        while id == None:
            (id, filepath, file) = random.choice(hit_index)
            if index_contains(case_index, id):
                id = None
        case_index.append((id, filepath, file))
        shutil.copyfile(filepath, f'{results_dir}{file}')
        print(f'Added ID {id}.')

def main():
    hit_index = []
    for i in range(1, 11):
        update_case_index(f'{dir_braid}results{i}/VGG19/photos_test/hit/', hit_index)

    print(f'Distinct hit cases: {len(hit_index)}.')

    for i in range(1, 11):
        extra_dir = f'{dir_braid}results{i}/VGG19/photos_test/extra/'
        if os.path.exists(extra_dir):
            shutil.rmtree(extra_dir)
        os.mkdir(extra_dir)
        
        case_index = []
        update_case_index(f'{dir_braid}results{i}/VGG19/photos_test/', case_index)
        missing_cnt = 364 - len(case_index)
        print(f'Experiment {i} is missing {missing_cnt} cases.')

        add_missing_cases(extra_dir, case_index, hit_index)

if __name__ == "__main__":
    main()
