import gc
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from sklearn.model_selection import ParameterGrid
import datetime
import os
import numpy as np
import h5py

from models import MDAD_model, single_MLP_model, single_linear_model
from experiment_helpers import load_data_for_fold, save_MTL_predictions
from configs import * 

os.environ["CUDA_VISIBLE_DEVICES"]="2"
K.tensorflow_backend._get_available_gpus()



# Point to where to save results/predictions/models
path_to_results = CV_save_path + "results/MTL/"
path_to_preds = CV_save_path + "predictions/MTL/"
path_to_models = CV_save_path + "models/MTL/"

hy_dict_list = list(ParameterGrid(hyperparams))

for fold_idx in range(30):

    print("FOLD:",fold_idx)

    X_train, X_valid, y_train, y_valid = load_data_for_fold(fold_idx)

    y_train_dict = {}
    y_valid_dict = {}
    for p in phenotypes:
        y_train_dict["%s_out"%p] = y_train[p]
        y_valid_dict["%s_out"%p] = y_valid[p]


    for hy_iteration in range(len(hy_dict_list)):

        print(datetime.datetime.now())

        print("HYPERPARAMETER ITERATION: %d"%hy_iteration)

        hy_dict = hy_dict_list[hy_iteration]

        title = "%d_%s_%s_%s_%f_%f_%f_%s_%f_%d"%(hy_dict["epochs"], hy_dict["nonlinearity"], 
                        str(hy_dict["hidden_sizes_shared"]), str(hy_dict["hidden_sizes_separate"]),
                        hy_dict["dropout"], hy_dict["k_reg"], hy_dict["learning_rate"], 
                        str(hy_dict["loss_weights"]),  hy_dict["grad_clip_norm"], hy_dict["batch_size"])
        print(title)

        res_dest = path_to_results + "/" + split_pca_dataset + "/" + title + "/"
        preds_dest = path_to_preds + "/" + split_pca_dataset + "/" + title + "/"
        modelpath =  path_to_models  + split_pca_dataset + "/" + title + "/" + str(fold_idx) + "/"

        for path in [res_dest, preds_dest, modelpath]:
            if not os.path.isdir(path):
                os.makedirs(path)


        model = MDAD_model(X_train, hy_dict)

        # https://stackoverflow.com/questions/36895627/python-keras-creating-a-callback-with-one-prediction-for-each-epoch
        class prediction_history(Callback):
            def __init__(self):
                self.predhis = []
            def on_epoch_end(self, epoch, logs={}):
                self.predhis.append(model.predict(X_valid))
        predictions=prediction_history()


        if fold_idx > 24:
            History = model.fit(x={'main_input': X_train}, y=y_train_dict, 
                      validation_data = ({'main_input': X_valid}, y_valid_dict),                                     
                      verbose=0,epochs=hy_dict["epochs"], batch_size=hy_dict["batch_size"], 
                    callbacks=[CSVLogger(res_dest+'%d.log'%fold_idx), predictions, 
                    # save model:
                    ModelCheckpoint(modelpath+"{epoch:02d}.hdf5", monitor='val_loss', verbose=0, \
                    save_best_only=False, save_weights_only=False, mode='auto', period=100)])
        else:
            History = model.fit(x={'main_input': X_train}, y=y_train_dict, 
                      validation_data = ({'main_input': X_valid}, y_valid_dict),    
                      verbose=0, epochs=hy_dict["epochs"], batch_size=hy_dict["batch_size"], 
                    callbacks=[CSVLogger(res_dest+'%d.log'%fold_idx), predictions])

        save_MTL_predictions(predictions, preds_dest, fold_idx, y_valid, phenotypes)

        K.clear_session()
        gc.collect()