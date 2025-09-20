import os
import pandas as pd
import matplotlib.pyplot as plt
import json

dir_braid = '/home/hicup/disk/braid/'
#dir_results = f'{dir_braid}results/'

photos_subfolder = 'photos_test'

full_confusion = {}
groups = ["113", "1211", "122", "11", "22", "111", "112", "1112", "12", "1111", "123", "1212", "1222"]

def process_folder(path, name, dir_results):
    global full_confusion

    matrices = {}
    correct = 0
    incorrect = 0

    cnt = 0
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                cnt += 1
                parts = file.split('.')[0].split('_')
                truth = parts[1]
                predicted = parts[2]

                if name == 'VGG19':
                    full_confusion[truth][predicted] += 1

                if not truth in matrices:
                    matrices[truth] = {'TP': 0, 'FP': 0, 'FN': 0}
                if not predicted in matrices:
                    matrices[predicted] = {'TP': 0, 'FP': 0, 'FN': 0}
                
                if truth == predicted:
                    correct += 1
                    matrices[truth]['TP'] += 1
                else:
                    incorrect += 1
                    matrices[predicted]['FP'] += 1
                    matrices[truth]['FN'] += 1

    #print(matrices.keys())

    #print(matrices)
    matrices = dict(sorted(matrices.items()))
    #print(matrices)

    plot_labels = []
    plot_precision = []
    plot_recall = []

    sum_precision = 0
    sum_recall = 0
    sum_f = 0

    metrics = {}
    for key in matrices:
        matrix = matrices[key]
        tp, fn, fp = matrix['TP'], matrix['FN'], matrix['FP']

        if tp + fp > 0:
            precision = 100 * (tp / (tp + fp))
        else:
            precision = 0

        if tp + fn > 0:
            recall = 100 * (tp / (tp + fn))
        else:
            recall = 0

        if precision + recall > 0:
            f = 2 * (precision * recall)/(precision + recall)
        else:
            f = 0

        sum_precision += precision
        sum_recall += recall
        sum_f += f

        plot_labels.append(key)
        plot_precision.append(precision)
        plot_recall.append(recall)

        metrics[key] = {'precision': precision, 'recall': recall, 'F1': f}

    # print(sum_precision/len(matrices), sum_recall/len(matrices))

    plot_data = pd.DataFrame(
        {'precision': plot_precision, 'recall': plot_recall},
        index = plot_labels
    )

    plot_data.plot(kind='bar', figsize=(20,4))
    plt.savefig(f'{dir_results}{name}/precision-recall.png')
    plt.close()

    return 100*correct/(correct+incorrect), sum_precision/len(matrices), sum_recall/len(matrices), sum_f/len(matrices), matrix

def process_results(number = None):
    if number is None:
        dir_results = f'{dir_braid}results/'
    else:
        dir_results = f'{dir_braid}results{number}/'

    fname = f'{dir_results}precision-recall.txt'

    # Delete existing results
    if os.path.exists(fname):
        os.remove(fname)
    
    results = {}

    for dir in os.listdir(dir_results):
        if os.path.isdir(f'{dir_results}{dir}') and os.path.exists(f'{dir_results}{dir}/{photos_subfolder}/'):
            ca, precision, recall, f, matrix = process_folder(f'{dir_results}{dir}/{photos_subfolder}/', dir, dir_results)
            results[dir] = {'matrix': matrix, 'CA':ca, 'precision': precision, 'recall': recall, 'F1': f}
            if number is None:
                print(dir, ca, precision, recall, f)
            else:
                print(number, dir, ca, precision, recall, f)

    with open(fname, "w") as outfile: 
        json.dump(results, outfile)

def main():
    global full_confusion, groups

    for truth in groups:
        full_confusion[truth] = {}
        for prediction in groups:
            full_confusion[truth][prediction] = 0

    process_results()

    #for i in range(10):
    #    process_results(i + 1)
    
    for truth in groups:
        for prediction in groups:
            print(full_confusion[truth][prediction], end=' ')
        print()

if __name__ == "__main__":
    main()
