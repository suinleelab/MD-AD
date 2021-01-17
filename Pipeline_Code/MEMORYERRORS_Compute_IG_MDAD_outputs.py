import gc
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras import optimizers, regularizers, losses

from keras.models import Model
from keras import backend as K
from keras.callbacks import CSVLogger
from keras import metrics

import scipy
import datetime 

import keras

import sys
import pickle


import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

sys.path.append("../packages/")
import IntegratedGradients as IG
from configs import * 
from models import * 


with h5py.File(path_to_MDAD_data_folders + "%s/%s.h5"%(SPECIFIC_FOLDER, full_pca_dataset), 'r') as hf:
    PCA_components = hf["PCA_components_"][:]
    gene_symbols = hf["gene_symbols"][:]
    

with h5py.File(path_to_MDAD_data_folders + "%s/%s.h5"%(SPECIFIC_FOLDER, full_dataset), 'r') as hf:
    raw_X = hf["ge"][:].astype(np.float64)      
    raw_Y = hf["labels"][:]
    raw_gene_symbols = hf["gene_symbols"][:]

    
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


def get_gene_weight_output(model, X):
    L = model.output.shape[1] 
        
    IG_L_by_N_by_G = np.zeros([L, len(X), len(gene_symbols)])

    ig = IG.integrated_gradients(model)

    for lvar in range(L):
        if lvar%10 == 0:
            print(lvar, datetime.datetime.now())
            
        IG_L_by_N_by_G[lvar] = np.array([ig.explain(x, outc=lvar) for x in X]) 
        
    return IG_L_by_N_by_G 



def get_gene_weight_latent(model, X, savepath):
    L = model.output.shape[1] 
        
    IG_L_by_N_by_G = np.zeros([L, len(X), len(gene_symbols)])

    ig = IG.integrated_gradients(model)
    
    non_triv_idx = np.where(np.sum(np.abs(model.predict(X)),axis=0) > 0)[0]
    
    print(non_triv_idx)
    for lvar in non_triv_idx:
        print(lvar, datetime.datetime.now())
            
        cur_weights =  np.array([ig.explain(x, outc=lvar) for x in X]) 
        
        if np.abs(cur_weights).sum() > 0:
            print("(nontriv)")
            if not os.path.isdir(savepath):
                os.makedirs(savepath)

            with h5py.File(savepath + "%i.h5"%(lvar), 'w') as hf:
                hf.create_dataset("gene_weights", data=cur_weights)

def get_gene_weight_latent_node(model, X, node):
    L = model.output.shape[1] 
        
    IG_L_by_N_by_G = np.zeros([L, len(X), len(gene_symbols)])

    ig = IG.integrated_gradients(model)
    
    node_weights =  np.array([ig.explain(x, outc=node) for x in X]) 
    return node_weights

                
def get_PCA_stacked_model(model, raw_X, method, fname):


    main_input = Input(shape=(raw_X.shape[1],), dtype='float', name='main_input')
    submean = Dense(raw_X.shape[1], activation="linear", name='submean')(main_input)
    pcatrans = Dense(num_components, activation="linear", name='pcatrans')(submean)
    model.layers.pop(0)
    out_model = model(pcatrans)

    if method == "MTL":
        MTL_phenotype_output_mapping = {"BRAAK":0, "CERAD":1, "PLAQUES":2, "TANGLES":3, "ABETA_IHC":4, "TAU_IHC":5}
        model_w_PCA = Model(inputs=[main_input], outputs=[out_model[MTL_phenotype_output_mapping[phenotype]]])
    else:
        model_w_PCA = Model(inputs=[main_input], outputs=[out_model])

    model_w_PCA.layers[1].set_weights([np.identity(raw_X.shape[1]), -1*raw_X.mean(axis=0)])
    model_w_PCA.layers[2].set_weights([PCA_components.T[:,:500], np.zeros(500)])


    grad_clip_norm = float(fname.split("_")[-2])
    learning_rate = float(fname.split("_")[-4])
    opt = optimizers.adam(clipnorm=grad_clip_norm, lr=learning_rate)  
    model_w_PCA.compile(optimizer=opt, loss = "mse")

    return model_w_PCA






MTL_final_final_model = pickle.load(open(path_to_final_models_chosen + "MTL/final.p", "rb" ) )
baselines_final_final_model = pickle.load(open(path_to_final_models_chosen + "MLP_baselines/final.p", "rb" ) )

method = "MTL"



      
path_to_centroid_info = "/media/big/nbbwang/md-ad_public_repo_data/Modeling/final_model/origGE/model_transformations_consensus/1/normed_KMeans_medoids/MTL_50_medoids_info.csv"
centroid_info = pd.read_csv(path_to_centroid_info).sort_values("cluster")

consenus_IG_weights = np.zeros([len(centroid_info), len(raw_X), len(gene_symbols)])

for i,row in centroid_info.iterrows():
    run=row["run"]
    node_idx=row["node_idx"]
    
    path_to_model = final_models_save_path + "models/MTL/ACT_MSBBRNA_ROSMAP_PCA/%s/%i/200.hdf5"%(MTL_final_final_model, run)
    MTL_up_to_latent = get_model_layers(path_to_model, 4)
    main_input = Input(shape=(raw_X.shape[1],), dtype='float', name='main_input')
    submean = Dense(raw_X.shape[1], activation="linear", name='submean')(main_input)
    pcatrans = Dense(500, activation="linear", name='pcatrans')(submean)
    MTL_up_to_latent.layers.pop(0)
    out_model = MTL_up_to_latent(pcatrans)
    model_w_PCA = Model(inputs=[main_input], outputs=[out_model])
    model_w_PCA.layers[1].set_weights([np.identity(raw_X.shape[1]), -1*raw_X.mean(axis=0)])
    model_w_PCA.layers[2].set_weights([PCA_components.T[:,:num_components], np.zeros(500)])
    grad_clip_norm = float(fname.split("_")[-2])
    learning_rate = float(fname.split("_")[-4])
    opt = optimizers.adam(clipnorm=grad_clip_norm, lr=learning_rate)  
    model_w_PCA.compile(optimizer=opt, loss = "mse")

    print("Computing IG for cluster node %i (run: %i, node from run: %i)"%(i, run, node_idx))

    consenus_IG_weights[i] = get_gene_weight_latent_node(model_w_PCA, raw_X, node_idx)
    
    savepath = "IG_weights_stacked2/consensus/%s/%s/%i/last_shared/"%(SPECIFIC_FOLDER, method)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    with h5py.File(savepath + "%i.h5"%(i), 'w') as hf:
        hf.create_dataset("gene_weights", data=consenus_IG_weights[i])
    
    K.clear_session()
    gc.collect()
    