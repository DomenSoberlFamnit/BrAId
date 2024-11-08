import os
import json
import h5py
from PIL import Image

cnt_all = 0
cnt_seen = 0
cnt_unchanged = 0
cnt_changed_ok = 0
cnt_groups_chg = 0

error_flags = ['yolo_error', 'photo_truncated', 'vehicle_joined', 'vehicle_split', 'cannot_label', 'inconsistent_data', 'off_lane', 'wrong_lane', 'multiple_vehicles', 'fixed']

def find_vehicle(rv, id):
    for data in rv:
        if str(data['photo_id']) == id:
            return data
    return None

def get_box(segments, color):
    for seg in segments:
        box = seg['box']
        if box['color'] == color:
            return (box['x'], box['y'], box['width'], box['height'])
    return None

def img_resize_224(img):
    ratio = img.width / img.height

    out = Image.new(mode="RGB", size=(224, 224), color="black")

    if ratio == 1:
        img1 = img.resize((224, 224))
        out.paste(img1)
    elif ratio > 1:
        img1 = img.resize((224, round(224/ratio)))
        out.paste(img1, (0, round((224 - img1.height)/2)))
    else:
        img1 = img.resize((round(224*ratio), 224))
        out.paste(img1, (round((224 - img1.width)/2), 0))

    return out

#def prop_has_errors(prop):
#    if 'errors' not in prop:
#        return False
#
#    errors = prop['errors']
#
#    for key in errors:
#        if errors[key] != 0:
#            return True
#
#    return False

def prop_has_errors(prop):
    if 'errors' not in prop:
        return False
    
    errors = prop['errors']

    for flag in error_flags:
        if flag in errors and errors[flag] != 0:
            return True

    return False

def get_photo(src_photos_dir, rv_record, id, seg):
    subdir = str(int(int(id)/1000)) + '/'
    try:
        img = Image.open(src_photos_dir + subdir + str(id) + '.png')
    except:
        return None

    (x, y, w, h) = get_box(rv_record['segments'], seg)
    cropped = img.crop((x, y, x+w, y+h))

    return img_resize_224(cropped)

def check_photo(file, rv_record, dir, id):
    global cnt_all, cnt_seen, cnt_unchanged, cnt_changed_ok

    prop = json.loads(file[f'{dir}/{id}'].asstr()[()])

    seg = prop['segment'] if 'segment' in prop else 'r'
    true_groups = prop['axle_groups'] if 'axle_groups' in prop else rv_record['axle_groups']

    cnt_all += 1
    if prop['seen_by'] != None:
        cnt_seen += 1
        if prop['changed_by'] == None:
            cnt_unchanged += 1
            return True, true_groups, seg
        else:
            if not prop_has_errors(prop):
                cnt_changed_ok += 1
                return True, true_groups, seg

    return False, true_groups, seg

def run(dir_braid):
    global cnt_all, cnt_seen, cnt_unchanged, cnt_changed_ok, cnt_groups_chg

    cnt_all = 0
    cnt_seen = 0
    cnt_unchanged = 0
    cnt_changed_ok = 0
    cnt_groups_chg = 0

    src_photos_dir = dir_braid + 'photos/'
    dst_photos_dir = dir_braid + 'cropped_photos/'

    if not os.path.exists(dst_photos_dir):
        os.mkdir(dst_photos_dir)

    f = open(dir_braid + 'recognized_vehicles.json')
    rv = json.load(f)
    f.close()

    with h5py.File('../metadata/metadata.hdf5', 'r') as file:
        for dir in file.keys():
            data = file[dir]
            for id in data:
                rv_record = find_vehicle(rv, id)
                if rv_record == None:
                    continue

                photo_ok, true_groups, seg = check_photo(file, rv_record, dir, id)
                if photo_ok:
                    if true_groups != dir:
                        cnt_groups_chg += 1

                    img = get_photo(src_photos_dir, rv_record, id, seg)
                    if img != None:
                        if not os.path.exists(dst_photos_dir + f'{true_groups}/'):
                            os.mkdir(dst_photos_dir + f'{true_groups}/')
                        img.save(dst_photos_dir + f'{true_groups}/{id}.png')
                        print(f'Cropping photo {cnt_all}/{len(rv)} with ID {id} of groups {dir} -> {true_groups}')

    print("All YOLO photos:", len(rv))
    print("Considered photos:", cnt_all)
    print("Photos seen by someone:", cnt_seen)
    print("Photos not changed:", cnt_unchanged)
    print("Photos changed but used:", cnt_changed_ok)
    print("Photos where groups are changed:", cnt_groups_chg)
