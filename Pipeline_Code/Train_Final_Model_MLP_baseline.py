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


from models import MDAD_model, single_MLP_model, single_linear_model
from experiment_helpers import load_final_PCA_data, save_MTL_predictions

from configs import * 

os.environ["CUDA_VISIBLE_DEVICES"]="2"
K.tensorflow_backend._get_available_gpus()


# Point to selected model hyperparameters
baselines_final_selected_model = pickle.load(open(path_to_final_models_chosen + "MLP_baselines/final.p", "rb" ) )

# Location to save models 
path_to_results = final_models_save_path + "results/MLP_baselines/"
path_to_models = final_models_save_path + "models/MLP_baselines/"

hy_dict_list = list(ParameterGrid(hyperparams))


for rep in [0]:
    X,y = load_final_PCA_data(SPECIFIC_FOLDER)

    for hy_iteration in range(len(hy_dict_list)):

        hy_dict = hy_dict_list[hy_iteration]

        title = "%d_%s_%s_%s_%f_%f_%f_%s_%f_%d"%(hy_dict["epochs"], hy_dict["nonlinearity"], 
                        str(hy_dict["hidden_sizes_shared"]), str(hy_dict["hidden_sizes_separate"]),
                        hy_dict["dropout"], hy_dict["k_reg"], hy_dict["learning_rate"], 
                        str(hy_dict["loss_weights"]),  hy_dict["grad_clip_norm"], hy_dict["batch_size"])
            
        for phenotype in  phenotypes:
            print(phenotype)
            if title == baselines_final_selected_model[phenotype]:

                model = single_MLP_model(X, hy_dict) 

                modelpath = path_to_models + full_pca_dataset + "/" + phenotype + "/" +title +  "/" + str(rep) + "/"
                res_dest = path_to_results +  "/" + full_pca_dataset + "/" + phenotype +"/" + title + "/"+ str(rep) + "/"
                for path in [modelpath, res_dest]:
                    if not os.path.isdir(path):
                        os.makedirs(path)

                csv_logger = CSVLogger(res_dest+'results.log')

                History = model.fit(x={'main_input': X},y={'out': y[phenotype]}, 
                                    verbose=0,epochs=hy_dict["epochs"], batch_size=hy_dict["batch_size"], callbacks=[csv_logger,\
                                    ModelCheckpoint(modelpath+"{epoch:02d}.hdf5", monitor='val_loss', verbose=1, \
                                    save_best_only=False, save_weights_only=False, mode='auto', period=200)])

                K.clear_session()
                gc.collect()   
