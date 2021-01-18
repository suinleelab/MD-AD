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
import pickle

from models import MDAD_model
from experiment_helpers import load_final_PCA_data, save_MTL_predictions

from configs import * 

os.environ["CUDA_VISIBLE_DEVICES"]="1"
K.tensorflow_backend._get_available_gpus()


# Point to selected model hyperparameters
MTL_final_final_model = pickle.load(open(path_to_final_models_chosen + "MTL/final.p", "rb" ) )

# Location to save models 
path_to_results = final_models_save_path + "results/MTL/"
path_to_models = final_models_save_path + "models/MTL/"

hy_dict_list = list(ParameterGrid(hyperparams))

for rep in range(100):
    print(rep)
    
    X,y = load_final_PCA_data(SPECIFIC_FOLDER)

    for hy_iteration in range(len(hy_dict_list)):
    
        hy_dict = hy_dict_list[hy_iteration]

        title = "%d_%s_%s_%s_%f_%f_%f_%s_%f_%d"%(hy_dict["epochs"], hy_dict["nonlinearity"], 
                        str(hy_dict["hidden_sizes_shared"]), str(hy_dict["hidden_sizes_separate"]),
                        hy_dict["dropout"], hy_dict["k_reg"], hy_dict["learning_rate"], 
                        str(hy_dict["loss_weights"]),  hy_dict["grad_clip_norm"], hy_dict["batch_size"])
        
        if title == MTL_final_final_model:
            print(title)

            res_dest = path_to_results +  "/" + full_pca_dataset + "/" + title + "/" + str(rep) + "/"
            modelpath = path_to_models + full_pca_dataset + "/" + title +  "/" + str(rep) + "/"
            for path in [res_dest, modelpath]:
                if not os.path.isdir(path):
                    os.makedirs(path)


            model = MDAD_model(X, hy_dict)    

            csv_logger = CSVLogger(res_dest+'results.log')

            History = model.fit(x={'main_input': X},\
                                y={'BRAAK_out': y["BRAAK"], 'CERAD_out': y["CERAD"], \
                                   'PLAQUES_out': y["PLAQUES"], 'TANGLES_out': y["TANGLES"],
                       "ABETA_IHC_out": y["ABETA_IHC"], "TAU_IHC_out":y["TAU_IHC"]}, \
                      verbose=0,epochs=hy_dict["epochs"], batch_size=hy_dict["batch_size"], callbacks=[csv_logger,\
                    ModelCheckpoint(modelpath+"{epoch:02d}.hdf5", monitor='val_loss', verbose=0, \
                    save_best_only=False, save_weights_only=False, mode='auto', period=200)])


            K.clear_session()
            gc.collect()