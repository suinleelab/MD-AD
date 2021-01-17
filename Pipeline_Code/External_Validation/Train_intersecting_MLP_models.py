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

os.environ["CUDA_VISIBLE_DEVICES"]="1"
K.tensorflow_backend._get_available_gpus()

hy_dict_list = list(ParameterGrid(hyperparams))
baselines_final_final_model = pickle.load(open(path_to_configs+path_to_final_models_chosen + "MLP_baselines/final.p", "rb" ) )


for dataset in ["Blood_GSE63060", "Blood_GSE63061", "Mouse"]:
    path_to_data = path_to_configs + path_to_ext_val_data_folder + "processed_intersection/%s/"%dataset
    path_to_all = path_to_configs + path_to_ext_val_results + "intersecting_gene_models/%s/"%dataset
    path_to_results = path_to_all + "results/MLP_baselines/"
    path_to_models = path_to_all + "models/MLP_baselines/"


    X,y=load_ext_val_intersection_data(path_to_data)


    for rep in range(10):
        print(rep)
        for hy_iteration in range(len(hy_dict_list)):
            hy_dict = hy_dict_list[hy_iteration]

            title = "%d_%s_%s_%s_%f_%f_%f_%s_%f_%d"%(hy_dict["epochs"], hy_dict["nonlinearity"], 
                            str(hy_dict["hidden_sizes_shared"]), str(hy_dict["hidden_sizes_separate"]),
                            hy_dict["dropout"], hy_dict["k_reg"], hy_dict["learning_rate"], 
                            str(hy_dict["loss_weights"]),  hy_dict["grad_clip_norm"], hy_dict["batch_size"])

            for phenotype in  ["ABETA_IHC", "TAU_IHC","PLAQUES", "TANGLES","BRAAK", "CERAD"]:

                if title == baselines_final_final_model[phenotype]:
                    print(phenotype)
                    print(datetime.datetime.now())
                    
                    model = single_MLP_model(X, hy_dict) 

                    modelpath = path_to_models + dataset + "/" + phenotype + "/" +title +  "/" + str(rep) + "/"
                    if not os.path.isdir(modelpath):
                        os.makedirs(modelpath)


                    res_dest = path_to_results +  "/"+dataset + "/"+ phenotype +"/" + title + "/"+ str(rep) + "/"
                    if not os.path.isdir(res_dest):
                        os.makedirs(res_dest)

                    csv_logger = CSVLogger(res_dest+'results.log')

                    History = model.fit(x={'main_input': X},y={'out': y[phenotype]}, 
                    verbose=0,epochs=hy_dict["epochs"], batch_size=hy_dict["batch_size"], callbacks=[csv_logger,\
                    keras.callbacks.ModelCheckpoint(modelpath+"{epoch:02d}.hdf5", monitor='val_loss', verbose=1, \
                         save_best_only=False, save_weights_only=False, mode='auto', period=200)])

                    K.clear_session()
                    gc.collect()

