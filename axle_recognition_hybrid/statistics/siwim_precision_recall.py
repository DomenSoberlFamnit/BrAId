import h5py
import json
import matplotlib.pyplot as plt
import pandas as pd

dir_braid = '/home/hicup/disk/braid/'

plot_classes = ['1111', '1112', '111', '22', '1222', '113', '123', '122', '1211', '11', '1212', '112', '12']
error_flags = ['yolo_error', 'photo_truncated', 'vehicle_joined', 'vehicle_split', 'cannot_label', 'inconsistent_data', 'off_lane', 'wrong_lane', 'multiple_vehicles']

def add_raised_axles(groups, raised_axles):
    raised_groups = list(groups)
    for axle in raised_axles.split(','):
        idx = int(axle) - 1
        raised_groups[idx] = str(int(raised_groups[idx]) + 1)
    return ''.join(raised_groups)

def prop_has_errors(prop):
    if 'errors' not in prop:
        return False
    
    errors = prop['errors']

    for flag in error_flags:
        if flag in errors and errors[flag] != 0:
            return True

    return False

def get_siwim_groups(filepath):
    hdf = pd.read_hdf(filepath)
    siwim_groups = {}
    for index, row in hdf.iterrows():
        id = row['id']
        if id == 'nan':
            continue
        rp1 = row['rp01_grp']
        rp2 = row['rp02_grp']
        rp3 = row['rp03_grp']
        rp2_fixed = row['rp02_fixed']
        rp3_fixed = row['rp03_fixed']
        siwim_groups[str(id)] = {'rp1': (rp1, False), 'rp2': (rp2, rp2_fixed), 'rp3': (rp3, rp3_fixed)}
    return siwim_groups

def compute_confusion_matrix(siwim_stage=None):
    siwim_groups = get_siwim_groups('../metadata/grp_and_fixed.hdf5')

    with h5py.File('../metadata/metadata.hdf5', 'r') as file:
        cnt = 0
        cnt_seen = 0
        cnt_changed = 0
        cnt_changed_ok = 0
        cnt_ok = 0
        cnt_raised_correct = 0

        cnt_siwim_groups_found = 0

        correct = 0
        tp = {}
        fn = {}
        fp = {}
        grp_size = {}

        for groups in file.keys():
            data = file[groups]
            for id in data:
                prop = json.loads(file[f'{groups}/{id}'].asstr()[()])
                true_groups = prop['axle_groups'] if 'axle_groups' in prop else groups
                predicted_groups = groups
                groups_fixed = 'errors' in prop and 'fixed' in prop['errors']

                cnt += 1
                if cnt % 10000 == 0:
                    print(f'Processed {cnt} records.')

                photo_ok = False

                if prop['seen_by'] == None:
                    continue

                cnt_seen += 1

                if prop['changed_by'] == None:
                    photo_ok = True
                else:
                    cnt_changed += 1
                    if not prop_has_errors(prop):
                        photo_ok = True
                        if true_groups != groups:
                            cnt_changed_ok += 1
                
                # Photo OK means that we include it into the database.
                if photo_ok:
                    cnt_ok += 1
                   
                    # Take the siwim prediction based on the given siwim siwim stage
                    if siwim_stage != None:
                        if id in siwim_groups:
                            cnt_siwim_groups_found += 1
                            (predicted_groups, groups_fixed) = siwim_groups[id][siwim_stage]

                    if true_groups not in grp_size:
                        grp_size[true_groups] = 0
                    grp_size[true_groups] += 1

                    # groups is what SiWIM detected.
                    # true_groups is the camera ground_truth.
                    siwim_correct = True
                    if predicted_groups != true_groups:
                        siwim_correct = False
                        # But check, if the groups changed because of the raised axles.
                        if 'raised_axles' in prop and len(prop['raised_axles'].strip()) > 0:
                            if add_raised_axles(groups, prop['raised_axles']) == true_groups:
                                siwim_correct = True
                                cnt_raised_correct += 1

                    if siwim_correct:
                        correct += 1
                        if true_groups not in tp:
                            tp[true_groups] = 0
                        tp[true_groups] += 1
                    else:
                        if groups not in fp:
                            fp[groups] = 0
                        fp[groups] += 1
                        if true_groups not in fn:
                            fn[true_groups] = 0
                        fn[true_groups] += 1

    n = cnt_ok
    
    print(f'Seen: {cnt_seen}, ok: {cnt_ok}, changed: {cnt_changed}, changed and ok {cnt_changed_ok}, correct because raised {cnt_raised_correct}, found in siwim database: {cnt_siwim_groups_found}.')
    print(f'Correct: {correct}/{n} ({100 * correct / n}%).')
    
    confusion_matrix = {}
    for groups in tp.keys():
        if grp_size[groups] > 0:
            if groups not in tp:
                tp[groups] = 0
            if groups not in fn:
                fn[groups] = 0
            if groups not in fp:
                fp[groups] = 0

            assert tp[groups] + fn[groups] == grp_size[groups]

            tn = n - tp[groups] - fn[groups] - fp[groups]
            confusion_matrix[groups] = {'TP': tp[groups], 'FN': fn[groups], 'FP': fp[groups], 'TN': tn}

    return dict(sorted(confusion_matrix.items()))

def compute_precision_recall(confusion_matrix):
    stat = {}

    plot_labels = []
    plot_precision = []
    plot_recall = []

    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for groups in confusion_matrix.keys():
        matrix = confusion_matrix[groups]
        tp = matrix['TP']
        fn = matrix['FN']
        fp = matrix['FP']
        tn = matrix['TN']
        n = tp + fn + fp + tn

        precision = 100 * (tp / (tp + fp)) if tp + fp > 0 else 0
        recall = 100 * (tp / (tp + fn)) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall)/(precision + recall) if precision + recall > 0 else 0

        weight = (tp + fn) / n
        total_precision += weight * precision
        total_recall += weight * recall
        total_f1 += weight * f1

        if groups in plot_classes:
            plot_labels.append(groups)
            plot_precision.append(precision)
            plot_recall.append(recall)

        stat[groups] = {'precision': precision, 'recall': recall, 'F1': f1}

    plot_data = pd.DataFrame(
        {'precision': plot_precision, 'recall': plot_recall},
        index = plot_labels
    )

    plot_data.plot(kind='bar', figsize=(20,6))
    plt.savefig(f'{dir_braid}siwim-precision-recall.png')

    n = len(confusion_matrix.keys())
    return stat, total_precision, total_recall, total_f1

def process_for_stage(siwim_stage=None):
    confusion_matrix = compute_confusion_matrix(siwim_stage)

    #with open(f'{dir_results}siwim-confusion_matrix.json', 'w') as file:
    #    json.dump(confusion_matrix, file)
    
    stat, precision, recall, f1 = compute_precision_recall(confusion_matrix)

    f = open(f'{dir_braid}siwim-precision-recall.txt', 'w')
    f.write(f'groups,precision,recall,F1\n')
    for groups in stat:
        values = stat[groups]
        f.write(f'{groups},{values['precision']},{values['recall']},{values['F1']}\n')
    f.write(f'AVERAGE,{precision},{recall},{f1}\n')
    f.close()

def main():
    print('Computing for stage RP1')
    process_for_stage('rp1')

    print('Computing for stage RP2')
    process_for_stage('rp2')
    
    print('Computing for stage RP3')
    process_for_stage('rp3')
    
    print('Computing for metadata')
    process_for_stage()

if __name__ == "__main__":
    main()
