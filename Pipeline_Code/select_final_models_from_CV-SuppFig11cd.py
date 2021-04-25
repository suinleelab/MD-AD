# PACKAGES
import numpy as np
import os
from scipy import stats as ss
import pickle 
import h5py
import pandas as pd
from configs import * 



path_to_results_folder = "%sresults/"%CV_save_path
path_to_preds_folder = "%spredictions/"%CV_save_path

path_to_final_chosen_models = "%sfinal_models_chosen_NCREV/"%CV_save_path
if  not os.path.isdir(path_to_final_chosen_models):
    os.makedirs(path_to_final_chosen_models)
    
    
#### Compute test set variances

variances = {}

tmp_fname = os.listdir(path_to_preds_folder + "MTL/" + split_pca_dataset + "/")[0]
for variable in phenotypes:
    vars_by_split = []
    for fold_idx in range(25,30):
        path_to_preds = path_to_preds_folder + "MTL/%s/%s/%d.h5"%(split_pca_dataset, tmp_fname, fold_idx)
        with h5py.File(path_to_preds, 'r') as hf:
            true_vals = hf["y_true"][variable][:]

        vars_by_split.append(np.nanvar(true_vals))    
    variances[variable] = np.array(vars_by_split)
    

def MTL_get_CV_test_res_as_df(path_to_log_files):
    
    firstfile = os.listdir(path_to_log_files)[0]
    cols = pd.read_csv(path_to_log_files + firstfile).columns

    test_runs = []
    for i in range(5):
        # the first 25 files are CV folds
        cur_idx = 25+i
        test_runs.append(pd.read_csv(path_to_log_files + "%d.log"%cur_idx))

    test_overall_averages = pd.DataFrame(np.nanmean(np.array([test_runs[i].values for i in range(5)]),axis=0), columns=cols)
    return test_runs, test_overall_averages


#################################################
############### MD-AD  ##########################
#################################################

performances = {}
for var in phenotypes:
    performances[var] = {}

    for fname in  os.listdir(path_to_results_folder + "MTL/" + split_pca_dataset):
        test_runs, test_overall_avergaes = MTL_get_CV_test_res_as_df(path_to_results_folder + "MTL/" + split_pca_dataset + "/" + fname + "/")
        performances[var][fname] = []

        for foldidx in range(5):
            # we want the min loss, and we scale by variance
            test_var = variances[var][foldidx]
            performances[var][fname].append(np.min(test_runs[foldidx]["val_%s_out_loss"%var])/test_var)
            
            
fnames =  os.listdir(path_to_results_folder + "MTL/" + split_pca_dataset)
num_hy = len(fnames)

############  Choose final model within each CV split ##################

FOLD_performances = {}
for fold_idx in range(5):
    FOLD_performances[fold_idx] = {}
    for key1 in performances.keys():
        FOLD_performances[fold_idx][key1] = []
        for key2 in performances[key1].keys():
            FOLD_performances[fold_idx][key1].append(performances[key1][key2][fold_idx])
            
            
FOLD_rankings = {}
for fold_idx in range(5):

    FOLD_rankings[fold_idx] = {}    
    for phenotype in FOLD_performances[fold_idx].keys():
        if np.sum(~np.isnan(FOLD_performances[fold_idx][phenotype])) == 0:
            FOLD_rankings[fold_idx][phenotype] = np.zeros(num_hy)
            continue
        
        FOLD_rankings[fold_idx][phenotype] = ss.rankdata(FOLD_performances[fold_idx][phenotype])

        
FOLD_sum_of_ranks = {}
for fold_idx in range(5):
    FOLD_sum_of_ranks[fold_idx] = np.zeros(num_hy)
    
    for phenotype in FOLD_rankings[fold_idx].keys():
        FOLD_sum_of_ranks[fold_idx] += FOLD_rankings[fold_idx][phenotype]

final_models_dict = {}
for fold_idx in range(5):
    final_models_dict[25+fold_idx] = fnames[np.argmin(FOLD_sum_of_ranks[fold_idx])]
    
    
if not os.path.isdir(path_to_final_chosen_models + "MTL/"):
    os.makedirs(path_to_final_chosen_models + "MTL/")

pickle.dump(final_models_dict, open( path_to_final_chosen_models+"MTL/folds.p", "wb" ) )



############  Choose final model overall (for retraining with all data) ##################

AVG_performances = {}
for key1 in performances.keys():
    AVG_performances[key1] = {}
    for key2 in performances[key1].keys():
        AVG_performances[key1][key2] = np.nanmean(performances[key1][key2])

fnames_performances = {}
for phenotype in AVG_performances.keys():
    fnames_performances[phenotype] = []
    for fname in fnames:
        fnames_performances[phenotype].append(AVG_performances[phenotype][fname])

fnames_rankings = {}
for phenotype in fnames_performances.keys():
    fnames_rankings[phenotype] = ss.rankdata(fnames_performances[phenotype])
    
sum_of_ranks = np.zeros(len(fnames))
for phenotype in fnames_rankings.keys():
    sum_of_ranks += fnames_rankings[phenotype]
    
pickle.dump(fnames[np.argmin(sum_of_ranks)], open( path_to_final_chosen_models+"MTL/final.p", "wb" ) )




#################################################
################ MLPs  ##########################
#################################################

for method in ["MLP_baselines", "Linear_baselines"]:
    performances = {}
    for var in phenotypes:

        performances[var] = {}

        for fname in  os.listdir(path_to_results_folder + method + "/" + split_pca_dataset):
            test_runs, test_overall_avergaes = MTL_get_CV_test_res_as_df("%s%s/%s/%s/%s/"%(path_to_results_folder,method,split_pca_dataset, fname,var))
            performances[var][fname] = []

            for foldidx in range(5):
                # we want the min loss, and we scale by variance
                test_performance = np.min(test_runs[foldidx]["val_loss"])/variances[var][foldidx]
                performances[var][fname].append(test_performance)
            
    ############  Choose final model within each CV split ##################

    FOLD_performances = {}
    for fold_idx in range(5):
        FOLD_performances[fold_idx] = {}
        for key1 in performances.keys():
            FOLD_performances[fold_idx][key1] = []
            for key2 in performances[key1].keys():
                FOLD_performances[fold_idx][key1].append(performances[key1][key2][fold_idx])


    FOLD_rankings = {}
    for fold_idx in range(5):

        FOLD_rankings[fold_idx] = {}

        for phenotype in FOLD_performances[fold_idx].keys():
            if np.sum(~np.isnan(FOLD_performances[fold_idx][phenotype])) == 0:
                FOLD_rankings[fold_idx][phenotype] = np.zeros(len(fnames))
                continue

            FOLD_rankings[fold_idx][phenotype] = ss.rankdata(FOLD_performances[fold_idx][phenotype])

    final_models = {}
    for fold_idx in range(5):
        final_models[25+fold_idx] = {}
        for phenotype in FOLD_rankings[fold_idx].keys():
            final_models[25+fold_idx][phenotype] = fnames[np.argmin(FOLD_rankings[fold_idx][phenotype])]

    if not os.path.isdir(path_to_final_chosen_models + method):
        os.makedirs(path_to_final_chosen_models + method)

    pickle.dump(final_models, open( path_to_final_chosen_models+"%s/folds.p"%method, "wb" ) )


    ############  Choose final model overall (for retraining with all data) ##################


    FOLD_sum_of_ranks = {}

    for fold_idx in range(5):
        FOLD_sum_of_ranks[fold_idx] = np.zeros(len(fnames))

        for phenotype in FOLD_rankings[fold_idx].keys():
            FOLD_sum_of_ranks[fold_idx] += FOLD_rankings[fold_idx][phenotype]


    AVG_performances = {}
    for key1 in performances.keys():
        AVG_performances[key1] = {}
        for key2 in performances[key1].keys():
            AVG_performances[key1][key2] = np.nanmean(performances[key1][key2])

    fnames_performances = {}
    for phenotype in AVG_performances.keys():
        fnames_performances[phenotype] = []
        for fname in fnames:
            fnames_performances[phenotype].append(AVG_performances[phenotype][fname])

    fnames_rankings = {}
    for phenotype in fnames_performances.keys():
        fnames_rankings[phenotype] = ss.rankdata(fnames_performances[phenotype])

    final_final_mlp_baselines = {}
    for phenotype in fnames_rankings.keys():
        final_final_mlp_baselines[phenotype]= fnames[np.argmin(fnames_rankings[phenotype])]

    pickle.dump(final_final_mlp_baselines, open( path_to_final_chosen_models+"%s/final.p"%method, "wb" ) )

    print("Saved selected hyperparameters to folder: %s"%path_to_final_chosen_models)