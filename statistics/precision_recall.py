import os
import pandas as pd
import matplotlib.pyplot as plt
import json

dir_braid = '/home/hicup/disk/braid/'
dir_results = f'{dir_braid}results/'

photos_subfolder = 'photos_test'

def process_folder(path, name):
    matrices = {}

    cnt = 0
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                cnt += 1
                parts = file.split('.')[0].split('_')
                truth = parts[1]
                predicted = parts[2]

                if not truth in matrices:
                    matrices[truth] = {'TP': 0, 'FP': 0, 'FN': 0}
                if not predicted in matrices:
                    matrices[predicted] = {'TP': 0, 'FP': 0, 'FN': 0}
                
                if truth == predicted:
                    matrices[truth]['TP'] += 1
                else:
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

        precision = 100 * (tp / (tp + fp))
        recall = 100 * (tp / (tp + fn))
        f = 2 * (precision * recall)/(precision + recall)

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

    return sum_precision/len(matrices), sum_recall/len(matrices), sum_f/len(matrices), matrix

def main():
    fname =f'{dir_results}precision-recall.txt'

    # Delete existing results
    if os.path.exists(fname):
        os.remove(fname)
    
    results = {}

    for dir in os.listdir(dir_results):
        if os.path.isdir(f'{dir_results}{dir}') and os.path.exists(f'{dir_results}{dir}/{photos_subfolder}/'):
            precision, recall, f, matrix = process_folder(f'{dir_results}{dir}/{photos_subfolder}/', dir)
            results[dir] = {'matrix': matrix, 'precision': precision, 'recall': recall, 'F1': f}
            print(dir, precision, recall, f)

    with open(fname, "w") as outfile: 
        json.dump(results, outfile)


if __name__ == "__main__":
    main()