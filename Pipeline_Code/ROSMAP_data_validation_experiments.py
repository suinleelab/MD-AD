import h5py
import pandas as pd
import numpy as np
from itertools import chain, combinations
import importlib
import ROSMAP_data_validation_experiments_helpers
from tensorflow.python.client import device_lib
from models import linear_baseline
import pickle
import pandas as pd
from configs import path_to_MDAD_data_folders, path_to

path_to_saved_results = "../../Pipeline_Outputs_Submitted/origGE/results/MTL/ROSMAP_CV_subsets_results.csv"

with h5py.File(path_to_MDAD_data_folders + "ACT_MSBBRNA_ROSMAP_PCA.h5") as hf:
    labels_names = hf["labels_names"][:]
    labels = hf["labels"][:]
    labels_df = pd.DataFrame(labels, columns=labels_names.astype(str))

variances = pd.read_csv(path_to_MDAD_data_folders + "test_set_variances.csv")


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item
            
def short_subset_name(subset):
    short_names = {"ACT.tsv": "A", "ROSMAP_GE1.tsv": "R", "MSBB_RNA.tsv": "M"}
    short_name = ""
    for item in subset:
        short_name += short_names[item]
    return short_name

train_datasets = ["ACT.tsv", "MSBB_RNA.tsv", "ROSMAP_GE1.tsv"]
test_datasets = ["ROSMAP_GE1.tsv"]
folds = range(25, 30)
subsets = powerset(train_datasets)


results_for_fold = {}

linear_results_df = pd.DataFrame(columns=['Val', 'Phenotype', 'Fold', 'Subset'])

for fold in folds:
    print(fold)
    for phenotype in ["CERAD", "PLAQUES", "ABETA_IHC", "BRAAK", "TANGLES", "TAU_IHC"]:
        print(phenotype)
        for subset in powerset(train_datasets):
            if not subset:
                continue
            importlib.reload(data_validation_experiments)
            test = ROSMAP_data_validation_experiments_helpers.DataValidationExperiment(fold, subset, test_datasets)
            test.run_experiment()
            X_train, X_valid, y_train, y_valid = test.load_data_for_fold(fold)
            
            indices_to_keep = np.where(np.isfinite(y_train[phenotype]))
            X_train = X_train[indices_to_keep]
            y_train[phenotype] = y_train[phenotype][indices_to_keep]
            if X_train.shape[0] > 0:
                
                indices_to_keep = np.where(np.isfinite(y_valid[phenotype]))
                X_valid = X_valid[indices_to_keep]
                y_valid[phenotype] = y_valid[phenotype][indices_to_keep]
                
                model = linear_baseline(
                    X_train, 
                    {
                        "learning_rate": 0.0001,
                        "k_reg": 0.001,
                        "grad_clip_norm": 0.1
                    })
                model.fit(x={'main_input': X_train},
                                y={'out': y_train[phenotype]}, 
                      validation_data = ({'main_input': X_valid}, 
                                         {'out' :y_valid[phenotype]}),                                     
                      verbose=0,epochs=200, batch_size=20)

                predictions = model.predict(X_valid)
                mse = ((predictions - y_valid[phenotype])**2).mean()
                mse /= variances.iloc[fold][phenotype]
            else:
                mse = 0
            linear_results_df = linear_results_df.append({
                "Val": mse, "Phenotype": phenotype, "Fold": fold, "Subset": short_subset_name(subset)
            }, ignore_index=True)
            
pickle.dump(linear_results_df, open("linear_results_df.p", "wb"))

variances = pd.read_csv("test_set_variances.csv")
subset_to_results = {}

def get_final_results(subset, fold_idx):
    subset_results = pd.read_csv(
        "MODEL_OUTPUTS/results/md-ad_data_validation/ACT_MSBBRNA_ROSMAP_PCASplit/" + str(subset) + "/" + str(fold_idx) + ".log"
    )
    filter_cols = [col for col in subset_results if col.startswith("val")]
    subset_results = subset_results[filter_cols]
    final_subset_results = subset_results.iloc[199]
    return final_subset_results

subsets = powerset(train_datasets)
mlp_results_df = pd.DataFrame(columns=['Val', 'Phenotype', 'Fold', 'Subset'])

datasets_to_phenotypes = {
    "A": ["ABETA_IHC", "TAU_IHC", "CERAD", "BRAAK"],
    "R": ["ABETA_IHC", "BRAAK", "CERAD", "PLAQUES", "TANGLES", "TAU_IHC"],
    "M": ["PLAQUES", "CERAD", "BRAAK"]
}

def phenotype_in_subset(phenotype, subset):
    for dataset in subset:
        if phenotype in datasets_to_phenotypes[dataset]:
            return True
    return False

for fold in folds:
    print(fold)
    for phenotype in ["CERAD", "PLAQUES", "ABETA_IHC", "BRAAK", "TANGLES", "TAU_IHC"]:
        print(phenotype)
        for subset in powerset(train_datasets):
            if not subset:
                continue
            final_subset_results_for_fold = get_final_results(subset, fold)
            val = final_subset_results_for_fold["val_{}_out_loss".format(phenotype)] / variances.iloc[fold][phenotype] if phenotype_in_subset(phenotype, short_subset_name(subset)) else 0
            
            mlp_results_df = mlp_results_df.append({
                "Val": val,
                "Phenotype": phenotype,
                "Fold": fold,
                "Subset": short_subset_name(subset)
            }, ignore_index=True)
            
mlp_results_df.to_csv(path_to_saved_results)