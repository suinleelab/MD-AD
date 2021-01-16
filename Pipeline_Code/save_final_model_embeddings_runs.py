import h5py 
import numpy as np
from keras import optimizers, regularizers, losses
from keras.models import Model
from keras import backend as K
import keras
import sys
import pickle
import tensorflow as tf
import os
import gc


from configs import * 
from models import ignorenans_categorical_accuracy, ordloss, ignorenans_mse, ignorenans_scaled_mse


os.environ["CUDA_VISIBLE_DEVICES"]="2"
K.tensorflow_backend._get_available_gpus()



with h5py.File(path_to_MDAD_data_folders + "%s.h5"%(full_pca_dataset), 'r') as hf:
    X = hf["ge_transformed"][:,:num_components].astype(np.float64)      
    

MTL_final_final_model = pickle.load(open(path_to_final_models_chosen + "MTL/final.p", "rb" ) )
baselines_final_final_model = pickle.load(open(path_to_final_models_chosen + "MLP_baselines/final.p", "rb" ) )


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


######### MLP ##########################################################

MLP_LAYER_IDXes = [2,4,6,8]


X_transformed_layers = {}

for phenotype in phenotypes:

    fname = baselines_final_final_model[phenotype]
    

    path_to_models = final_models_save_path+ "models/MLP_baselines/%s/%s/%s/"%("ACT_MSBBRNA_ROSMAP_PCA", phenotype, fname)


    repfolders_idx = np.where(["." not in x for x in  os.listdir(path_to_models)])[0]
    repfolders = np.array(os.listdir(path_to_models))[repfolders_idx]

    # from a single run, it's obvious that MLP is overfitting to the phenotypes, so no need for consensus calculations
    repfolders = ["0"]
    for rep in repfolders:
        path_to_model = path_to_models + "%s/200.hdf5"%rep

        full_model = keras.models.load_model(path_to_model, custom_objects={"ordloss_cur_params": ordloss(0), \
            "ignorenans_mse": ignorenans_mse, "cat_acc": ignorenans_categorical_accuracy(0), \
            "ignorenans_scaled_mse": ignorenans_scaled_mse})

        for HIDDEN_LAYER in [1]:

            if not os.path.isdir("%sMLP/%s/%i/"%(final_rep_embeddings_savepath, phenotype, HIDDEN_LAYER)):
                os.makedirs("%sMLP/%s/%i/"%(final_rep_embeddings_savepath, phenotype, HIDDEN_LAYER))

            # GET TRANSFORMATIONS FOR THE MODEL UP TO THE LAYER OF INTEREST 
            mod_to_layer = get_model_layers(path_to_model, MLP_LAYER_IDXes[HIDDEN_LAYER])

            X_transformed = mod_to_layer.predict(X)
            
            print(phenotype, HIDDEN_LAYER, rep)
        
            np.savetxt("%sMLP/%s/%i/%s.txt"%(final_rep_embeddings_savepath, phenotype, HIDDEN_LAYER, rep),X_transformed)

            
            
######### MD-AD ##########################################################

MTL_LAYER_IDXes = [1, 3,\
                  {"BRAAK": 5, "CERAD":6, "PLAQUES": 7, "TANGLES":8, "ABETA_IHC":9, "TAU_IHC":10}, 
                  {"BRAAK": 17, "CERAD":18, "PLAQUES": 19, "TANGLES":20, "ABETA_IHC":21, "TAU_IHC":22},
                  {"BRAAK": 29, "CERAD":30, "PLAQUES": 31, "TANGLES":32, "ABETA_IHC":33, "TAU_IHC":34}]



MTL_final_final_model

path_to_models = final_models_save_path + "models/MTL/%s/%s/"%(full_pca_dataset, MTL_final_final_model)


repfolders_idx = np.where(["." not in x for x in  os.listdir(path_to_models)])[0]
repfolders = np.array(os.listdir(path_to_models))[repfolders_idx]

for rep in repfolders[60:]:
    path_to_model = path_to_models + "%s/200.hdf5"%rep


    for HIDDEN_LAYER in [1]:

        # GET TRANSFORMATIONS FOR THE MODEL UP TO THE LAYER OF INTEREST 
        if type(MTL_LAYER_IDXes[HIDDEN_LAYER]) == dict:

            for phen in phenotypes:

                mod_to_layer = get_model_layers(path_to_model, MTL_LAYER_IDXes[HIDDEN_LAYER][phen]+1)
                X_transformed = mod_to_layer.predict(X)
                print(HIDDEN_LAYER, phen, rep)
                if not os.path.isdir("%sMTL/%i/%s/"%(final_rep_embeddings_savepath, HIDDEN_LAYER, phen)):
                    os.makedirs("%sMTL/%i/%s/"%(final_rep_embeddings_savepath, HIDDEN_LAYER, phen))
                np.savetxt("%sMTL/%i/%s/%s.txt"%(final_rep_embeddings_savepath, HIDDEN_LAYER, phen, rep),X_transformed)
        else:
            mod_to_layer = get_model_layers(path_to_model, MTL_LAYER_IDXes[HIDDEN_LAYER]+1)
            X_transformed = mod_to_layer.predict(X)
            
            print(HIDDEN_LAYER, rep)
            if not os.path.isdir("%sMTL/%i/"%(final_rep_embeddings_savepath, HIDDEN_LAYER)):
                os.makedirs("%sMTL/%i/"%(final_rep_embeddings_savepath, HIDDEN_LAYER))
            np.savetxt("%sMTL/%i/%s.txt"%(final_rep_embeddings_savepath, HIDDEN_LAYER, rep),X_transformed)
            
        K.clear_session()
        gc.collect()
            
            
