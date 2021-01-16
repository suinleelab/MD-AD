

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

# Chooses which pre-processed data set is used. "origGE" refers to the data with no covariate correction for GE data
SPECIFIC_FOLDER = "origGE"



phenotypes = ["ABETA_IHC", "TAU_IHC","PLAQUES", "TANGLES","BRAAK", "CERAD"]

num_cats = {"CERAD": 4, "BRAAK": 6}

num_components = 500



CV_save_path = "../../md-ad_public_repo_data/Modeling/%s/"%SPECIFIC_FOLDER
CV_new_save_path = "../../md-ad_public_repo_data/Modeling/%s/"%SPECIFIC_FOLDER




final_models_save_path =  "../../md-ad_public_repo_data/Modeling/final_model/%s/"%SPECIFIC_FOLDER
path_to_final_models_chosen = "../../AD_Project/analyses/MTL_variable_tasks/6vars-continuous/%s/final_models_chosen/"%SPECIFIC_FOLDER
final_rep_embeddings_savepath = "../../md-ad_public_repo_data/Modeling/final_model/%s/model_transformations/"%SPECIFIC_FOLDER
final_rep_consensus_embeddings_savepath = "../../md-ad_public_repo_data/Modeling/final_model/%s/model_transformations_consensus/"%SPECIFIC_FOLDER


IG_save_path = "../../md-ad_public_repo_data/Modeling/IG_weights/"





#################  Submitted pipeline  ######################################################

CV_save_path = "../../Pipeline_Outputs_Submitted/%s/"%SPECIFIC_FOLDER
CV_new_save_path = "../../md-ad_public_repo_data/Modeling/%s/"%SPECIFIC_FOLDER


#path_to_models = "../../AD_Project/analyses/MTL_variable_tasks/6vars-continuous/%s/models/"%SPECIFIC_FOLDER
#path_to_models = "../../Pipeline_Outputs_Submitted/%s/models/"%SPECIFIC_FOLDER


############################################ DATA ############################################

full_dataset = "ACT_MSBBRNA_ROSMAP"
full_pca_dataset = "ACT_MSBBRNA_ROSMAP_PCA"
split_pca_dataset = "ACT_MSBBRNA_ROSMAP_PCASplit"

path_to_MDAD_data_folders = "../../md-ad_public_repo_data/DATA/MTL_data/"

