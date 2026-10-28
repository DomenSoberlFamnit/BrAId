import argparse
import os

import numpy as np

DIR_BRAID = '/mnt/workspace/braid/workdir_single/'
WORKING_DIR = DIR_BRAID
SIGNAL_LENGTH = 1280
TRAIN_DATA_PATH = os.path.join(WORKING_DIR, 'train_data.npy')
MODEL_CHOICES = ('tcn', 'cnn')


def fixed_data_path(working_dir, name):
    return os.path.join(working_dir, name)


def default_model_path(model_name=None, dir_braid=DIR_BRAID):
    if model_name is None:
        raise ValueError(f'Model not specified. Available models: {MODEL_CHOICES}')
    return os.path.join(dir_braid, f'{model_name}_model.keras')


def generate_data():
    import dataset

    vehicle_info = dataset.load_vehicle_info(DIR_BRAID, f'{DIR_BRAID}camera/')
    if vehicle_info is None:
        raise FileNotFoundError('Cannot load vehicle info.')

    data_train = dataset.get_data(
        dir_braid=DIR_BRAID,
        input_signal='11admp',
        output_type='pulses',
        normalized_signals=True,
        include_correct=True,
        include_fixed=True,
        signal_length=SIGNAL_LENGTH,
        shuffle=False,
        gen_daily_dist=False,
        exclude_days=['2014-4-3'],
    )

    data_test = dataset.get_data(
        dir_braid=DIR_BRAID,
        input_signal='11admp',
        output_type='pulses',
        normalized_signals=True,
        include_correct=True,
        include_fixed=True,
        signal_length=SIGNAL_LENGTH,
        shuffle=False,
        gen_daily_dist=False,
        include_days=['2014-4-3'],
    )

    (meta_train, X_train, Y_train) = dataset.add_channel(data_train)
    (meta_test, X_test, Y_test) = dataset.add_channel(data_test)

    print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
    return vehicle_info, meta_train, X_train, Y_train, meta_test, X_test, Y_test


def save_dataset(file_path, data):
    meta, X, Y = data
    np.save(file_path, {'meta': meta, 'X': X, 'Y': Y}, allow_pickle=True)
    print(f'Dataset saved to {file_path}')


def load_dataset(file_path):
    payload = np.load(file_path, allow_pickle=True).item()
    return payload['meta'], payload['X'], payload['Y']


def prepare_data(train_path=None, test_path=None, working_dir=WORKING_DIR):
    if train_path is None:
        train_path = fixed_data_path(working_dir, 'train_data.npy')
    if test_path is None:
        test_path = fixed_data_path(working_dir, 'test_data.npy')

    _, meta_train, X_train, Y_train, meta_test, X_test, Y_test = generate_data()
    train_data = (meta_train, X_train, Y_train)
    test_data = (meta_test, X_test, Y_test)
    save_dataset(train_path, train_data)
    save_dataset(test_path, test_data)
    return train_path, test_path


def build_tcn_model():
    import tensorflow as tf
    import tcn

    model = tcn.model(
        SIGNAL_LENGTH,
        tcn_filters=64,
        tcn_dilations=[1, 2, 4, 8, 16, 32, 64],
        dropout=0.2,
    )

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    return model


def build_cnn_model():
    import tensorflow as tf
    import cnn

    model = cnn.model(
        SIGNAL_LENGTH,
        cnn_filters=32,
        cnn_layers=(9, 7, 5),
        dropout=0.2,
    )

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    return model


def build_model(model_name=None):
    builders = {
        'tcn': build_tcn_model,
        'cnn': build_cnn_model,
    }
    if model_name is None:
        raise ValueError(f'Model not specified. Available models: {sorted(builders)}')
    if model_name not in builders:
        raise ValueError(f'Unsupported model: {model_name}. Available models: {sorted(builders)}')
    return builders[model_name]()


def train_model(model_name=None, epochs=100, batch_size=32, working_dir=WORKING_DIR):
    import numpy as np
    import tensorflow as tf

    if train_data_path is None:
        train_data_path = fixed_data_path(working_dir, 'train_data.npy')
    if model_path is None:
        model_path = default_model_path(model_name, working_dir)

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f'Train dataset not found: {train_data_path}')

    _, X_train, Y_train = load_dataset(train_data_path)

    class_weights = {0.0: 1.0, 1.0: 282.0}
    train_sample_weight = np.where(
        Y_train == 1,
        class_weights[1.0],
        class_weights[0.0],
    ).astype(np.float32)

    model = build_model(model_name=model_name)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
    )

    model.fit(
        x=X_train,
        y=Y_train,
        sample_weight=train_sample_weight,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        shuffle=True,
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(model_path)
    print(f'Model saved to {model_path}')
    return model


def evaluate_model(model_name=None, batch_size=32, class_threshold=0.5, working_dir=WORKING_DIR):
    import numpy as np
    import tensorflow as tf

    assert model_name is not None, 'Model name must be specified for evaluation.'

    test_data_path = fixed_data_path(working_dir, 'test_data.npy')
    model_path = default_model_path(model_name, working_dir)

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f'Test dataset not found: {test_data_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    import evaluation

    _, X_test, Y_test = load_dataset(test_data_path)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(X_test, batch_size=batch_size)

    precision, recall, f1_score, mae, tp, fn, fp, axles = evaluation.evaluate_predictions(
        predictions=predictions,
        pulses=Y_test,
        class_threshold=class_threshold,
        filter_kernel_size=9,
        tolerance=3,
        progress=True,
    )

    pr_auc, best_threshold = evaluation.compute_pr_auc(
        predictions=predictions,
        pulses=Y_test,
        filter_kernel_size=9,
        tolerance=3,
        model_name=model_name,
        output_dir=working_dir,
    )

    print('Test set:')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, MAE: {mae:.4f}')
    print(f'PR AUC: {pr_auc:.4f}, Best Threshold: {best_threshold:.4f}')
    print(f'Axles: {axles}, TP: {tp}, FN: {fn}, FP: {fp}')
    return precision, recall, f1_score, mae, tp, fn, fp, axles


def clean_files(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            print(f'Removed {path}')
        else:
            print(f'Not found: {path}')


def cross_validate_model(train_path=None, model_name=None, folds=5, epochs=100, batch_size=32, class_threshold=0.5, working_dir=WORKING_DIR):
    import tensorflow as tf
    import dataset
    import evaluation

    if train_path is None:
        train_path = fixed_data_path(working_dir, 'train_data.npy')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f'Train dataset not found: {train_path}')
    if folds < 2:
        raise ValueError('Cross validation requires at least 2 folds.')

    meta, X, Y = load_dataset(train_path)
    results = {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mae': [],
    }

    for fold_k in range(folds):
        (_, X_train, Y_train), (_, X_val, Y_val) = dataset.split_fold((meta, X, Y), fold_k, folds)

        class_weights = {0.0: 1.0, 1.0: 282.0}
        train_sample_weight = np.where(
            Y_train == 1,
            class_weights[1.0],
            class_weights[0.0],
        ).astype(np.float32)

        model = build_model(model_name=model_name)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
        )

        model.fit(
            x=X_train,
            y=Y_train,
            sample_weight=train_sample_weight,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stop],
            verbose=0,
        )

        predictions = model.predict(X_val, batch_size=batch_size)
        precision, recall, f1_score, mae, _, _, _, _ = evaluation.evaluate_predictions(
            pulses=Y_val,
            predictions=predictions,
            class_threshold=class_threshold,
            filter_kernel_size=9,
            tolerance=3,
            progress=False,
        )

        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1_score)
        results['mae'].append(mae)

        print(f'Fold {fold_k + 1}/{folds}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}, MAE={mae:.4f}')

    print(f'Cross-validation summary ({folds} folds):')
    for metric_name, values in results.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        print(f'{metric_name}: {mean_value:.4f} ± {std_value:.4f}')

    return results


def build_parser():
    parser = argparse.ArgumentParser(
        description='Prepare train/test data, train the TCN model, or evaluate a saved model.',
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-data', action='store_true', help='Prepare the train/test datasets and save them as NumPy files.')
    mode.add_argument('--train', action='store_true', help='Train the TCN model from the saved training NumPy dataset.')
    mode.add_argument('--eval', action='store_true', help='Evaluate a saved model using the saved test NumPy dataset.')
    mode.add_argument('--cv', action='store_true', help='Run k-fold cross-validation on the training set only.')
    mode.add_argument('--clean', action='store_true', help='Delete generated data files and saved model artifacts.')
    parser.add_argument(
        '--working-dir',
        type=str,
        default=DIR_BRAID,
        help='Directory used for fixed-name dataset files and model artifacts.',
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=MODEL_CHOICES,
        help='Model architecture to train or evaluate. Available: tcn, cnn.',
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to the model file used for training or evaluation. Defaults to <model>_model.keras in DIR_BRAID.',
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and inference.')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument(
        '--class-threshold',
        type=float,
        default=0.5,
        help='Classification threshold used when converting predictions to binary pulses during evaluation.',
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.prepare_data and not args.train and not args.eval and not args.cv and not args.clean:
        parser.print_help()
        return 0

    working_dir = args.working_dir
    train_data_path = fixed_data_path(working_dir, 'train_data.npy')
    test_data_path = fixed_data_path(working_dir, 'test_data.npy')

    if args.clean:
        model_path = args.weights if args.weights is not None else default_model_path(args.model, working_dir) if args.model is not None else None
        clean_files([train_data_path, test_data_path] + ([model_path] if model_path is not None else []))
        return 0

    if args.model is None:
        print(f'Model not specified. Available models: {MODEL_CHOICES}')
        return 1

    effective_weights = args.weights if args.weights is not None else default_model_path(args.model, working_dir)

    if args.prepare_data:
        prepare_data(train_path=train_data_path, test_path=test_data_path, working_dir=working_dir)
        return 0

    if args.cv:
        cross_validate_model(
            model_name=args.model,
            folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_threshold=args.class_threshold,
            working_dir=working_dir,
        )
        return 0

    if args.train:
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            working_dir=working_dir,
        )
        return 0

    if args.eval:
        evaluate_model(
            model_name=args.model,
            batch_size=args.batch_size,
            class_threshold=args.class_threshold,
            working_dir=working_dir,
        )
        return 0

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

