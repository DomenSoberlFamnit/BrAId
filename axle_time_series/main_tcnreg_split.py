import dataset
from models.tcnreg import TCNReg

dir_braid = '/mnt/workspace/braid/workdir/'
signal_length = 1300

force_training = True

#####################   Data preparation   ########################################

vehicle_info = dataset.load_vehicle_info(dir_braid, f'{dir_braid}camera/')
if vehicle_info is None:
    print('Cannot load vehicle info.')
    quit()

data_train = dataset.get_data(
    dir_braid=dir_braid,
    input_signal='11admp',
    output_type='distances',
    normalized_signals=True,
    include_correct=True,
    include_fixed=False,
    signal_length=signal_length,
    shuffle=False,
    exclude_days=['2014-4-3']
)

#data_test = dataset.get_data(
#    dir_braid=dir_braid,
#    input_signal='11admp',
#    output_type='distances',
#    normalized_signals=True,
#    include_correct=False,
#    include_fixed=True,
#    signal_length=signal_length,
#    shuffle=False,
#    exclude_days=['2014-4-3']
#)

(meta_train, X_train, Y_train) = data_train
#(meta_test, X_test, Y_test) = data_test

###########################################   Train the model   #############################################

model = TCNReg(dir_braid, signal_length, tcn_filters=64, tcn_dilations=[1, 2, 3, 4, 5, 6, 7], dropout=0.1, y_neurons=256, s_neurons=64)
model.train(X_train, Y_train)

########################################   Evaluate the data   ########################################

#(predictions, tp, fn, fp, cnt_correct, cnt) = model.evaluate(X_test, Y_test, meta_test, class_threshold=0.2, kernel_size=9, vehicle_info=vehicle_info)
(predictions, tp, fn, fp, cnt_correct, cnt) = model.evaluate(X_train, Y_train, meta_train, kernel_size=9, vehicle_info=vehicle_info)
#model.classification_accuracy(kernel_size=9)