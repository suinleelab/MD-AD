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
from configs import * 
from models import * 

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)



with h5py.File(path_to_MDAD_data_folders + "%s.h5"%(full_pca_dataset), 'r') as hf:
    X = hf["ge_transformed"][:,:num_components].astype(np.float64)      
    Y = hf["labels"][:]    
    PCA_components = hf["PCA_components_"][:]
    gene_symbols = hf["gene_symbols"][:]
    labels_names = hf["labels_names"][:]

labels = pd.DataFrame(Y.astype(str), columns=labels_names.astype(str), dtype=str)

method = "MTL"
MTL_final_final_model = pickle.load(open(path_to_final_models_chosen + "MTL/final.p", "rb" ) )



#################### Get predictions for each run ##############################

MTL_phenotype_output_mapping = {}

method = "MTL"
for i in range(100):
    print(i)
    fname = MTL_final_final_model
    path_to_model = final_models_save_path + "models/MTL/ACT_MSBBRNA_ROSMAP_PCA/%s/%i/200.hdf5"%(fname,i)

    model = keras.models.load_model(path_to_model, custom_objects={"ordloss_cur_params": ordloss(0), \
        "ignorenans_mse": ignorenans_mse, "cat_acc": ignorenans_categorical_accuracy(0), \
        "ignorenans_scaled_mse": ignorenans_scaled_mse})

    for j,layer in enumerate(model.layers[-len(phenotypes):]):
        MTL_phenotype_output_mapping[layer.name[:-4]] = j

    preds = model.predict(X)

    if not os.path.isdir(path_to_preds):
        os.makedirs(path_to_preds)

    with h5py.File("%s%i.hdf5"%(path_to_preds,i), 'w') as hf:
        for phenotype in ["CERAD", "BRAAK", "PLAQUES", "TANGLES", "ABETA_IHC", "TAU_IHC"]:
            hf.create_dataset(phenotype, data=preds[MTL_phenotype_output_mapping[phenotype]])
    
    print("%s%i.hdf5"%(path_to_preds,i))
    K.clear_session()
    gc.collect()

    
###################### Get consensus prediction (average over runs)###########################

phenotypes = ['ABETA_IHC', 'BRAAK', 'CERAD', 'PLAQUES', 'TANGLES', 'TAU_IHC']
preds_runs = []
for f in os.listdir(path_to_preds):
    if f.split(".")[0].isnumeric():
        print(f)
        preds =[]
        for phenotype in phenotypes:
            with h5py.File(path_to_preds + "%i.hdf5"%i, 'r') as hf:
                preds.append(hf[phenotype][:])
        preds_runs.append(np.array(preds))
        
        
to_save_df = pd.DataFrame(np.mean(np.mean(np.array(preds_runs),axis=3),axis=0).T, columns=phenotypes)
to_save_df["sample_name"]  = labels["sample_name"]
to_save_df.to_csv(path_to_preds + "CONSENSUS_PREDICTIONS.csv")