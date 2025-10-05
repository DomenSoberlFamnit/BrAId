groups = ["113", "1211", "122", "11", "22", "111", "112", "1112", "12", "1111", "123", "1212", "1222"]

def print_matrix(matrix):
    #groups = []
    #for truth in matrix.keys():
    #    if truth not in groups:
    #        groups.append(truth)
    #        for prediciton in matrix[truth].keys():
    #            if prediciton not in groups:
    #                groups.append(prediciton)

    print('     ', end='')
    for group in groups:
        print(f'{group:<5}', end='')
    print()

    for truth in groups:
        sum = 0
        print(f'{truth:<5}', end='')
        for prediction in groups:
            if truth in matrix and prediction in matrix[truth]:
                sum += matrix[truth][prediction]
                print(f'{matrix[truth][prediction]:<5}', end='')
            else:
                print(f'{0:<5}', end='')
        print(f'{sum:<4}')

def print_errors(matrix, threshold):
    for truth in groups:
        for prediction in groups:
            if truth == prediction:
                continue
            if truth in matrix and prediction in matrix[truth]:
                cnt = matrix[truth][prediction]
                if cnt >= threshold:
                    print(f'{truth} -> {prediction}: {cnt}')

def process_file(filename):
    confusion_siwim = {}
    confusion_nn = {}

    print(filename)
    f = open(filename, 'r')

    raised_cnt = 0
    errors = {}

    for line in f:
        cols = line.strip().split(',')
        siwim = cols[1]
        nn = cols[8]
        camera = cols[2]
        road = cols[3]
        raised = cols[5]

        if raised == '1' and False:
            raised_cnt += 1
            continue

        # road = camera

        if road not in confusion_siwim:
            confusion_siwim[road] = {}
        
        if siwim not in confusion_siwim[road]:
            confusion_siwim[road][siwim] = 0
        
        confusion_siwim[road][siwim] += 1

        if camera not in confusion_nn:
            confusion_nn[camera] = {}
        
        if nn not in confusion_nn[camera]:
            confusion_nn[camera][nn] = 0
        
        confusion_nn[camera][nn] += 1

        if camera != nn:
            if camera not in errors:
                errors[camera] = 0
            errors[camera] += 1

    print_matrix(confusion_siwim)
    print()

    print_matrix(confusion_nn)
    print()
    print_errors(confusion_nn, 5)

    print("Raised:", raised_cnt)

    f.close()

files = ['stat-vgg16.csv', 'stat-vgg19.csv', 'stat-densenet.csv', 'stat-mobilenet.csv', 'stat-resnet.csv']

for file in files:
    process_file(file)
    print("---------------------------------")
