import gc
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras import optimizers, regularizers, losses
from keras import backend as K
from keras.models import Model
import pickle
import h5py
import keras
import sys
from sklearn.model_selection import ParameterGrid
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

K.tensorflow_backend._get_available_gpus()

from configs import * 
from models import ignorenans_categorical_accuracy, ordloss, ignorenans_mse, ignorenans_scaled_mse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



path_to_split_data = path_to_MDAD_data_folders + "%s"%(split_pca_dataset)

MTL_FINAL_MODELS = pickle.load(open(path_to_final_models_chosen + "MTL/folds.p", "rb" ) )
MLP_BASELINES_FINAL_MODELS =pickle.load(open(path_to_final_models_chosen + "MLP_baselines/folds.p", "rb" ) )
layer_number = 3

def get_model_layers(model_file, num_layers):
    
    # note: need to define custom functions for model in order to load, but these don't actually get used
    model = keras.models.load_model(model_file, custom_objects={"ordloss_cur_params": ordloss(0), \
            "ignorenans_mse": ignorenans_mse, "cat_acc": ignorenans_categorical_accuracy(0), \
            "ignorenans_scaled_mse": ignorenans_scaled_mse})
    
    # define new model that cuts off the last several layers
    newmodel = Model(inputs = model.input, outputs = model.layers[num_layers-1].output)
    
    # agian, need to specify these parameters, but they aren't used since we don't retrain the model
    opt = optimizers.adam()  
    newmodel.compile(optimizer=opt, loss= "mse")
    
    return newmodel


# get centroids for a new set of points
def kmeans_centroids_for_test(X_test, cluster_labels):
    n_clusters = len(np.unique(cluster_labels))
    n,d = X_test.shape
    
    new_centroids = np.zeros([n_clusters,n])
    for i in range(n_clusters):
        new_centroids[i] = np.mean(X_test.T[np.where(cluster_labels ==i)], axis=0).reshape([1,-1])

    return new_centroids


for mode in ["TEST", "TRAIN"]:

    
    for fold_idx in range(25,30):
        
        with h5py.File(path_to_split_data + "/" + str(fold_idx) + ".h5", 'r') as hf:
            X_train = hf["X_train_transformed"][:,:num_components].astype(np.float64)
            X_valid = hf["X_valid_transformed"][:,:num_components].astype(np.float64)
            labels_train = hf["y_train"][:]
            labels_valid = hf["y_valid"][:]
            labels_names = hf["labels_names"][:]

            
        ############### MD-AD MODEL ##################################################    

        if mode == "TEST":
            path_to_new_files = "%slast_shared_layer_transformations/MTL/"%(CV_save_path)
        else:
            path_to_new_files = "%slast_shared_layer_transformations_TRAIN/MTL/"%(CV_save_path)    

        hy_name = MTL_FINAL_MODELS[fold_idx]

        model = get_model_layers(CV_save_path + "models/MTL/ACT_MSBBRNA_ROSMAP_PCASplit/%s/%i/%i.hdf5"%(hy_name, fold_idx, 200), 4)

        print(path_to_new_files+ "%s/%i.h5"%(hy_name, fold_idx))
        if not os.path.isdir(path_to_new_files+ hy_name + "/"):
            os.makedirs(path_to_new_files+ hy_name + "/")

        with h5py.File(path_to_new_files + "%s/%i.h5"%(hy_name, fold_idx), 'w') as hf:
            if mode=="TEST":
                hf.create_dataset("labels", data=labels_valid)
                hf.create_dataset("outputs", data=model.predict(X_valid))
            else:
                hf.create_dataset("labels", data=labels_train)
                hf.create_dataset("outputs", data=model.predict(X_train))
            hf.create_dataset("labels_names", data=labels_names)

        ################## MLP MODELS ################################################

        if mode == "TEST":
            path_to_new_files = "%slast_shared_layer_transformations/MLP_baselines/"%(CV_save_path)
        else:
            path_to_new_files = "%slast_shared_layer_transformations_TRAIN/MLP_baselines/"%(CV_save_path)

        for phenotype in ["ABETA_IHC", "TAU_IHC", "CERAD", "BRAAK", "PLAQUES", "TANGLES"]:  

            MLP_final_path = CV_save_path + "models/MLP_baselines/" + split_pca_dataset + "/"

            hy_name = MLP_BASELINES_FINAL_MODELS[fold_idx][phenotype]

            model = get_model_layers(MLP_final_path + "%s/%s/%i/%i.hdf5"%(hy_name, phenotype, fold_idx, 200), 4)

            print(path_to_new_files+ "%s/%s/%i.h5"%(hy_name, phenotype, fold_idx))
            if not os.path.isdir(path_to_new_files+ hy_name + "/" + phenotype):
                os.makedirs(path_to_new_files+ hy_name + "/" + phenotype)

            with h5py.File(path_to_new_files + "%s/%s/%i.h5"%(hy_name, phenotype, fold_idx), 'w') as hf:
                if mode=="TEST":
                    hf.create_dataset("labels", data=labels_valid)
                    hf.create_dataset("outputs", data=model.predict(X_valid))
                else:
                    hf.create_dataset("labels", data=labels_train)
                    hf.create_dataset("outputs", data=model.predict(X_train))
                hf.create_dataset("labels_names", data=labels_names)

        K.clear_session()
        gc.collect()
        
        ############## UNSUPERVISED METHODS ###########################################

        if mode =="TRAIN":
            path_to_new_files = "%slast_shared_layer_transformations_TRAIN/unsupervised_methods/"%(CV_save_path)
        else:
            path_to_new_files = "%slast_shared_layer_transformations/unsupervised_methods/"%(CV_save_path)


        with h5py.File(path_to_split_data + "/%i.h5"%fold_idx, 'r') as hf:
            if mode == "TRAIN":
                X = hf["X_train_transformed"][:,:num_components].astype(np.float64)
                labels = hf["y_train"][:]
            else:
                X = hf["X_valid_transformed"][:,:num_components].astype(np.float64)
                labels = hf["y_valid"][:]

            X_train =  hf["X_train_transformed"][:,:num_components].astype(np.float64)
            gene_symbols = hf["gene_symbols"][:]
            labels_names = hf["labels_names"][:]

        for transformation in ["KMeans", "PCA"]:

            num_dims = 100

            print(path_to_new_files+ "%s/%i.h5"%(transformation, fold_idx))
            if not os.path.isdir(path_to_new_files + transformation + "/"):
                os.makedirs(path_to_new_files + transformation + "/")

            if transformation == "KMeans":
                kmeans = KMeans(n_clusters=num_dims).fit(X_train.T)
                X_transformed = kmeans_centroids_for_test(X, kmeans.labels_).T

            elif transformation == "PCA":
                pca = PCA(n_components=num_dims)
                pca.fit(X_train)
                X_transformed = pca.transform(X)[:, :num_dims]


            print(labels.shape, X_transformed.shape)
            with h5py.File(path_to_new_files + "%s/%i.h5"%(transformation, fold_idx), 'w') as hf:
                hf.create_dataset("labels", data=labels)
                hf.create_dataset("outputs", data=X_transformed)
                hf.create_dataset("labels_names", data=labels_names)