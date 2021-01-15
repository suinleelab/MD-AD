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

hy_dict_list = list(ParameterGrid(hyperparams))

for baseline_method in ["MLP_baselines", "Linear_baselines"]:
    print(baseline_method)
    
    # Point to where to save results/predictions/models
    path_to_results = CV_save_path + "results/%s/"%baseline_method
    path_to_preds = CV_save_path + "predictions/%s/"%baseline_method
    path_to_models = CV_save_path + "models/%s/"%baseline_method
    
    for fold_idx in range(30):
        
        print("FOLD:",fold_idx)
        X_train, X_valid, y_train, y_valid = load_data_for_fold(fold_idx)

        for hy_iteration in range(len(hy_dict_list)):

            print(datetime.datetime.now())

            print("HYPERPARAMETER ITERATION: %d"%hy_iteration)

            hy_dict = hy_dict_list[hy_iteration]

            title = "%d_%s_%s_%s_%f_%f_%f_%s_%f_%d"%(hy_dict["epochs"], hy_dict["nonlinearity"], 
                            str(hy_dict["hidden_sizes_shared"]), str(hy_dict["hidden_sizes_separate"]),
                            hy_dict["dropout"], hy_dict["k_reg"], hy_dict["learning_rate"], 
                            str(hy_dict["loss_weights"]),  hy_dict["grad_clip_norm"], hy_dict["batch_size"])
            print(title)
            for phenotype in phenotypes:

                print(phenotype)
                print(datetime.datetime.now())
    
                res_dest = path_to_results + "/" + split_pca_dataset + "/" + title + "/" + phenotype + "/"
                preds_dest = path_to_preds + "/" + split_pca_dataset + "/" + title + "/" 
                modelpath =  path_to_models  + split_pca_dataset + "/" + title + "/" + phenotype + "/" + str(fold_idx) + "/"

                for path in [res_dest, preds_dest, modelpath]:
                    if not os.path.isdir(path):
                        os.makedirs(path)           
                  
                
                if baseline_method == "MLP_baselines":
                    model = single_MLP_model(X_train, hy_dict)                                        
                else:
                    model = single_linear_model(X_train, hy_dict)
                
                print(X_train.shape, X_valid.shape)
                print(y_train[phenotype].shape, y_valid[phenotype].shape)


                # https://stackoverflow.com/questions/36895627/python-keras-creating-a-callback-with-one-prediction-for-each-epoch
                class prediction_history(Callback):
                    def __init__(self):
                        self.predhis = []
                    def on_epoch_end(self, epoch, logs={}):
                        self.predhis.append(model.predict(X_valid))

                predictions=prediction_history()


                csv_logger = CSVLogger(res_dest+ "/" + phenotype + "/" + '%d.log'%fold_idx)
                # And trained it via:
                if fold_idx > 24:
                    History = model.fit(x={'main_input': X_train}, y=y_train[phenotype], 
                              validation_data = ({'main_input': X_valid}, y_valid[phenotype]),                                     
                              verbose=0,epochs=hy_dict["epochs"], batch_size=hy_dict["batch_size"], 
                            callbacks=[CSVLogger(res_dest+'%d.log'%fold_idx), predictions, 
                            # save model:
                            ModelCheckpoint(modelpath+"{epoch:02d}.hdf5", monitor='val_loss', verbose=0, \
                            save_best_only=False, save_weights_only=False, mode='auto', period=100)])
                else:
                    History = model.fit(x={'main_input': X_train}, y=y_train[phenotype], 
                              validation_data = ({'main_input': X_valid}, y_valid[phenotype]),    
                              verbose=0, epochs=hy_dict["epochs"], batch_size=hy_dict["batch_size"], 
                            callbacks=[CSVLogger(res_dest+'%d.log'%fold_idx), predictions])

                    
                # SAVE PREDICTIONS
                with h5py.File(preds_dest + "%d.h5"%fold_idx, 'a') as hf:
                    # loop through epochs -- one group is made per epoch
                    for i, ep in enumerate(predictions.predhis):
                        if "/%s/%s"%(str(i),phenotype) in hf:
                            del hf["/%s/%s"%(str(i),phenotype)]
                        hf.create_dataset("/%s/%s"%(str(i),phenotype), data=predictions.predhis[i], dtype=np.float32)
                    if "/y_true/"+phenotype in hf:
                        del hf["/y_true/"+phenotype]
                    hf.create_dataset("/y_true/"+phenotype, data=y_valid[phenotype], dtype=np.float32)


                K.clear_session()
                gc.collect()