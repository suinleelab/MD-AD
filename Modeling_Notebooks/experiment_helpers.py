import numpy as np
import pandas as pd
import h5py


path_to_data_folders = "../../AD_Project/analyses/MTL_variable_tasks/MTL_data/"

def load_data_for_fold(fold_idx):
    # Hard-coded variables specific to MD-AD paper
    phenotypes = ["CERAD", "PLAQUES", "ABETA_IHC", "BRAAK", "TANGLES", "TAU_IHC"]
    num_components = 500
    num_cats = {"CERAD": 4, "BRAAK": 6}
    data_form = "ACT_MSBBRNA_ROSMAP_PCASplit"
    path_to_split_data = path_to_data_folders + data_form

    # ------------ LOAD DATA ---------------------------------------------------- #
    with h5py.File(path_to_split_data + "/" + str(fold_idx) + ".h5", 'r') as hf:
        if "PCA" in data_form:
            X_train = hf["X_train_transformed"][:, :num_components].astype(np.float64)
            X_valid = hf["X_valid_transformed"][:, :num_components].astype(np.float64)
        else:
            X_train = hf["X_train"][:].astype(np.float64)
            X_valid = hf["X_valid"][:].astype(np.float64)
        labels_train = hf["y_train"][:]
        labels_valid = hf["y_valid"][:]
        gene_symbols = hf["gene_symbols"][:]
        labels_names = hf["labels_names"][:]

    # ------------ PROCESS DATA FOR MODEL: TRAIN/TEST SETS + LABELS----------------- #
    labels_df_train = pd.DataFrame(labels_train, columns=labels_names.astype(str))
    labels_df_valid = pd.DataFrame(labels_valid, columns=labels_names.astype(str))

    shuffle_idx_train = np.random.permutation(range(len(labels_df_train)))
    shuffle_idx_valid = np.random.permutation(range(len(labels_df_valid)))

    X_train = X_train[shuffle_idx_train]
    X_valid = X_valid[shuffle_idx_valid]

    labels_df_train = labels_df_train.loc[shuffle_idx_train]
    labels_df_valid = labels_df_valid.loc[shuffle_idx_valid]

    y_train = {}
    y_valid = {}
    for phen in phenotypes:
        if phen in num_cats.keys():
            y_train[phen] = labels_df_train[phen].astype(float).values / (num_cats[phen] - 1)
            y_valid[phen] = labels_df_valid[phen].astype(float).values / (num_cats[phen] - 1)
        else:
            y_train[phen] = labels_df_train[phen].astype(float).values
            y_valid[phen] = labels_df_valid[phen].astype(float).values

    return X_train, X_valid, y_train, y_valid



def load_final_PCA_data(SPECIFIC_FOLDER):
    # Hard-coded variables specific to MD-AD paper
    phenotypes = ["CERAD", "PLAQUES", "ABETA_IHC", "BRAAK", "TANGLES", "TAU_IHC"]
    num_components = 500
    num_cats = {"CERAD": 4, "BRAAK": 6}
    dataset = "ACT_MSBBRNA_ROSMAP_PCA"

    # ------------ LOAD DATA ---------------------------------------------------- #
    with h5py.File("%s%s/%s.h5"%(path_to_data_folders, SPECIFIC_FOLDER,dataset), 'r') as hf:
        X = hf["ge_transformed"][:,:num_components].astype(np.float64)      
        Y = hf["labels"][:]
        labels_names = hf["labels_names"][:]

    # ------------ PROCESS DATA FOR MODEL---------------------------------------- #
    
    labels_df = pd.DataFrame(Y, columns=labels_names.astype(str))
    shuffle_idx = np.random.permutation(range(len(labels_df)))
    X = X[shuffle_idx]
    labels_df = labels_df.loc[shuffle_idx]
    
    y = {}
    for phen in phenotypes:
        if phen in num_cats.keys():
            y[phen] = labels_df[phen].astype(float).values / (num_cats[phen] - 1)
        else:
            y[phen] = labels_df[phen].astype(float).values
            
    return X, y




def save_MTL_predictions(predictions, preds_dest, fold_idx, y_valid, phenotypes):
    with h5py.File(preds_dest + "%d.h5"%fold_idx, 'w') as hf:
        # loop through epochs -- one group is made per epoch
        for i, ep in enumerate(predictions.predhis):
            # within each group created for each epoch, we save a dataset of predictions for the validation set
            for j, phenotype in enumerate(phenotypes):
                if "/%s/%s"%(str(i),phenotype) in hf:
                    del hf["/%s/%s"%(str(i),phenotype)]
                hf.create_dataset("/%s/%s"%(str(i),phenotype), data=predictions.predhis[i][j], dtype=np.float32)

        # save true values to /y_true/phenotype
        for phenotype in phenotypes:
            if "/y_true/"+phenotype in hf:
                del hf["/y_true/"+phenotype]
            hf.create_dataset("/y_true/"+phenotype, data=y_valid[phenotype], dtype=np.float32)