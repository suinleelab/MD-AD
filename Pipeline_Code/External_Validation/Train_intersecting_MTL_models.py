import gc
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from sklearn.model_selection import ParameterGrid
import keras
import datetime
import os
import numpy as np
import h5py
import pickle
import sys
path_to_configs = "../"
sys.path.append(path_to_configs)
from configs import *
from models import *
from experiment_helpers import load_ext_val_intersection_data

os.environ["CUDA_VISIBLE_DEVICES"]="0"
K.tensorflow_backend._get_available_gpus()

hy_dict_list = list(ParameterGrid(hyperparams))
MTL_final_final_model = pickle.load(open(path_to_configs+path_to_final_models_chosen + "MTL/final.p", "rb" ) )




for dataset in ["Blood_GSE63060", "Blood_GSE63061", "Mouse"]:
    print("Training models for %s"%dataset)
    
    path_to_data = path_to_configs + path_to_ext_val_data_folder + "processed_intersection/%s/"%dataset
    path_to_all = path_to_configs + path_to_ext_val_results + "intersecting_gene_models/%s/"%dataset
    path_to_results = path_to_all + "results/MTL/"
    path_to_models = path_to_all + "models/MTL/"


    X,y=load_ext_val_intersection_data(path_to_data)

    for rep in range(100):
        print("REP %i"%rep)

        for hy_iteration in range(len(hy_dict_list)):

            hy_dict = hy_dict_list[hy_iteration]

            title = "%d_%s_%s_%s_%f_%f_%f_%s_%f_%d"%(hy_dict["epochs"], hy_dict["nonlinearity"], 
                            str(hy_dict["hidden_sizes_shared"]), str(hy_dict["hidden_sizes_separate"]),
                            hy_dict["dropout"], hy_dict["k_reg"], hy_dict["learning_rate"], 
                            str(hy_dict["loss_weights"]),  hy_dict["grad_clip_norm"], hy_dict["batch_size"])

            if title == MTL_final_final_model:
                print(title)


                res_dest = path_to_results  + "/" + title + "/" + str(rep) + "/"
                if not os.path.isdir(res_dest):
                    os.makedirs(res_dest)

                model = MDAD_model(X, hy_dict)    

                modelpath = path_to_models  + "/" + title +  "/" + str(rep) + "/"
                if not os.path.isdir(modelpath):
                    os.makedirs(modelpath)


                csv_logger = CSVLogger(res_dest+'results.log')

                History = model.fit(x={'main_input': X},y={'BRAAK_out': y["BRAAK"], 'CERAD_out': y["CERAD"], \
                  'PLAQUES_out': y["PLAQUES"], 'TANGLES_out': y["TANGLES"],
                  "ABETA_IHC_out": y["ABETA_IHC"], "TAU_IHC_out":y["TAU_IHC"]}, \
                  verbose=0,epochs=hy_dict["epochs"], batch_size=hy_dict["batch_size"], callbacks=[csv_logger,\
                   keras.callbacks.ModelCheckpoint(modelpath+"{epoch:02d}.hdf5", monitor='val_loss', verbose=0, \
                        save_best_only=False, save_weights_only=False, mode='auto', period=200)])


                K.clear_session()
                gc.collect()
