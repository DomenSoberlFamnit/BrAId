import dataset
from tcn import TCNMult

dir_braid='/mnt/workspace/braid/workdir/'

data = dataset.load_or_generate(
    dir_braid,
    include_non_fixed=True,
    include_fixed=True,
    exclude_days=None,
    force_dataset_generation=False
)

data = dataset.shuffle(data)

fold_n = 10
results = []

for fold_k in range(fold_n):
    data_train, data_test = dataset.fold(data, fold_k, fold_n)

    (meta_train, (X_single_train, X_multiple_train), Y_train) = data_train
    (meta_test, (X_single_test, X_multiple_test), Y_test) = data_test

    model = TCNMult(dir_braid, signal_count=14, tcn_filters=64, tcn_dilations=[1, 2, 3, 4, 5, 6, 7], dropout=0.1)

    model.train(X=X_multiple_test, Y=Y_test)
    model.evaluate(X=X_multiple_test, Y=Y_test, meta=meta_test, class_threshold=0.5, kernel_size=9)

    quit()
