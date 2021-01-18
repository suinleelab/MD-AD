

# Hyperparameters to evaluate:  For each, hyperparameter, provide a LIST of options to evaluate. A paramter grid will be generated so all combinations of these hyperparameters will be evaluated.
hyperparams = {"epochs": [200], \
               "nonlinearity": ["relu"], \
               "hidden_sizes_shared": [[500,100]], \
               "hidden_sizes_separate": [[50,10]],\
               "dropout":  [.1],\
               "k_reg": [.00001,.001],\
               "learning_rate": [.0001,.001],\
               "loss_weights":  [[1, 1]],\
               "grad_clip_norm": [.01,.1],\
               "batch_size": [20]}

############################################ DATA ############################################

# Chooses which pre-processed data set is used. "origGE" refers to the data with no covariate correction for GE data
SPECIFIC_FOLDER = "origGE"

phenotypes = ["ABETA_IHC", "TAU_IHC","PLAQUES", "TANGLES","BRAAK", "CERAD"]
num_cats = {"CERAD": 4, "BRAAK": 6}
num_components = 500

full_dataset = "ACT_MSBBRNA_ROSMAP"
full_pca_dataset = "ACT_MSBBRNA_ROSMAP_PCA"
split_pca_dataset = "ACT_MSBBRNA_ROSMAP_PCASplit"

path_to_MDAD_data_folders = "../../DATA/MTL_data/"
path_to_ext_val_data_folder = "../../DATA/External_Validation/"
path_to_geneset_data = "../../DATA/geneset_data/"

###################### RESULTS ##################################

# Paths for saving cross-validation models and results 
CV_save_path = "../../Pipeline_Outputs_Submitted/%s/"%SPECIFIC_FOLDER
path_to_final_chosen_models = '../../Pipeline_Outputs_Submitted/%s/final_models_chosen/'%SPECIFIC_FOLDER

# path for final models and predictions
final_models_save_path = '../../Pipeline_Outputs_Submitted/final_model/%s/'%SPECIFIC_FOLDER
path_to_preds = "../../Pipeline_Outputs_Submitted/final_model/MDAD_predictions/"

# path for final embeddings
final_rep_embeddings_savepath = "../../Pipeline_Outputs_Submitted/model_transformations/"
final_rep_consensus_embeddings_savepath = "../../Pipeline_Outputs_Submitted/model_transformations_consensus/"
path_to_medoids_info = "../../Pipeline_Outputs_Submitted/model_transformations_consensus/1/normed_KMeans_medoids/MTL_50_medoids_info.csv"

# path for all external validaton results
path_to_ext_val_results =  "../../Pipeline_Outputs_Submitted/External_Validation/"

# path for model interpretations
IG_save_path = "../../Pipeline_Outputs_Submitted/IG_weights/"
path_to_gene_rankings = "../../Pipeline_Outputs_Submitted/gene_rankings/"
