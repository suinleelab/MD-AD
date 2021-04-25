# MD-AD

Code for paper "Unified AI framework to uncover deep interrelationships between gene expression and Alzheimer’s disease neuropathologies". Please read our preprint at the following link: https://doi.org/10.1101/2020.11.30.404087


## Code Overview

### Pipeline_Code

This folder contains all code used to generate all models and raw results.
The scripts for CROSS-VALIDATION and FINAL MODELS below should be run in the order listed: 

#### CROSS-VALIDATION:  Train and evaluate MD-AD models
- Training 5x5 models:
	- **_Train_CV_Models-MDAD.py_**
	- **_Train_CV_Models-Baselines.py_** (linear and MLP models)
	- These scripts each also save CV performance metrics which are displayed in Figure 2a.
- Evaluating prediction performance:
	- **_select_final_models_from_CV.py_** (based on CV performance, selects hyper parameters for the final models)
	- **_ROSMAP_data_validation_experiments.py_** (Evaluates just ROSMAP test set performance when training with different subsets of training data sets; Figure 2b)
- Compute last shared layer embeddings:
	- **_save_CV_model_embeddings.py_**  (Computes each splits’ final model’s last shared layer embedding for the train and test set. Models: MD-AD, MLPs, Unsupervised baselines)
	- These embeddings are evaluated in _Paper_Analyses/Evaluate_Embedding_Correlations.ipynb_ and _Paper_Analyses/t-SNE_plots_internal.ipynb_

#### FINAL MODELS (trained on full dataset)
- **_Train_Final_Model_MLP_baseline.py_**
- **_Train_Final_Model_MDAD.py_** (trains over multiple runs which will be used for consensus model)
- **_save_MDAD_final_predictions.py_** (saves model predictions for the full dataset over all runs; then saves a “consensus prediction” by averaging predictions across runs)
- **_save_final_model_embeddings_runs.py_** (saves model last shared layer embeddings for the full dataset over all runs. Also saves the analogous layer embedding of the MLPs)
- **_calculate_consensus_MD-AD_embedding.py_** (Generates a final consensus MD-AD embedding: (1) K-means clustering over all last shared layer node embeddings across runs, (2) for each cluster, identifies the nearest node to the center, (3) these centroids form the new embedding.)

Both the **External_Validation** and **Model_Interpretation** scripts below rely on completion of the above scripts, but they can be run independently of each other.

####  External_Validation [folder]
For new unseen datasets, we evaluate MD-AD with two possible approaches: (1) If many of the genes are common between the external and original datasets, we impute missing genes (using linear regression trained on available genes) and then directly apply the final MD-AD model on the new samples. (2) If many genes are missing, we instead train a new MD-AD model on intersecting genes (and relevant baselines) and then evaluate the new custom model on the external dataset.  
- Option 1:  Direct application of MD-AD model (human brain samples):
	- **_Save_transfer_predictions_runs.ipynb_**
- Option 2: Training new models with intersecting genes (mouse brain samples, human blood samples):
	- **_Train_intersecting_MTL_models.py_**
	- **_Train_intersecting_MLP_models.py_**
	- **_Save_intersecting_model_predictions_runs.ipynb_**
- For both approaches above:
	- Generate linear model predictions: **_linear_model_ext_val.ipynb_**
	- Generate embeddings for the MD-AD model: **_save_MDAD_embeddings_ext_val.ipynb_**
	- Generate consensus predictions for the MD-AD model: **_compute_final_model_predictions.ipynb_**
	- Does some cleaning of results files to make plotting easier: **_save_cleaned_results_for_plots.ipynb_**

#### Model_Interpretation [folder]
This folder contains code to extract Integrated Gradients values for each gene on each prediction. These scores are then used to rank genes according to their relevance to MD-AD’s pathology predictions. We also use GSEA to examine the enriched gene sets in the final model rankings.

- **_Get_MDAD_IG_weights.ipynb_** - Computes IG weights for each MD-AD run. For each run, generates a (# samples x # genes) matrix of IG values for each of the 6 phenotype outputs
- **_IG_weights_averaging.ipynb_** - Performs two kinds of averaging over the gene weights above:
	- Averaging over samples: For each run, we obtain average gene scores for each phenotype (#runs x #samples x #genes x #phenotypes) --> (#runs x #genes x #phenotypes)
	- Averaging over runs and phenotypes, so for each sample, we obtain a gene importance score: (#runs x #samples x #genes x #phenotypes) --> (#samples x #genes)
- **_Get_ranked_MDAD_genes.ipynb_** - Generates consensus gene scores across runs to rank the relative impact of each gene on predicted neuropathology severity
- **_Get_ranked_correlation_based_genes.ipynb_** - Baseline comparison approach for the MD-AD ranking. We obtain correlation-based gene rankings by averaging over the correlation coefficients between genes and each phenotype
- **_Run_GSEA.ipynb_** - Based on both MD-AD and correlation-based rankings, we check for the enrichment of gene sets in the rankings.

### Paper_Analyses
These notebooks process results from the MD-AD pipeline and generate figures presented in the paper:
- Figure 2a - Paper_Analyses/CV_Prediction_Performance.ipynb
- Figure 2b - Paper_Analyses/Subsets_CV_ROSMAP_Plots.ipynb
- Figures 2c-d, 7a-b - Paper_Analyses/External_validation_predictions.ipynb
- Figures 3a-c - Paper_Analyses/Evaluate_Embedding_Correlations.ipynb and Paper_Analyses/t-SNE_plots_internal.ipynb
- Figures 3d-e, 7c - Paper_Analyses/t-SNE_embeddings_external.ipynb
- Figure 4a-b, 6c - Paper_Analyses/Final_genes_color_by_genesets.ipynb
- Figure 4c - Paper_Analyses/Final_genes_rank_comparisons.ipynb
- Figures 5 and 6a-b - Paper_Analyses/Final_genes_interaction_analyses.ipynb

------------
### Dependencies 

This software was originally designed and run on a system running Ubuntu 16.04.3 with Python 3.3.6. For neural network model training and interpretation, we used a single Nvidia GeForce GTX 980 Ti GPU, though we anticipate that other GPUs will also work. Standard python software packages used: Tensorflow (1.3.0), Keras (2.0.4), numpy (1.17.3), pandas (0.24.1), scipy (1.3.1), scikit-learn (0.21.3), matplotlib (3.1.2), seaborn (0.9.0), h5py (2.9.0). We additionally used the following Python software packages available here: [IntegratedGradients](https://github.com/hiranumn/IntegratedGradients), and [GSEApy](https://pypi.org/project/gseapy/). 

-----------
## Data Availability:
**Gene expression and phenotype data:** The results published here are in based on data obtained from the [AD Knowledge Portal](https://adknowledgeportal.synapse.org/). Postmortem brain gene expression samples and phenotype labels are available from the AD Knowledge Portal for the following data sets (with listed synapse.org Synapse IDs):  ACT (syn5759376), ROSMAP (syn3219045), MSBB (RNA Sequencing: syn3159438, Microarray: syn3157699), Mayo Clinic Brain Bank (syn5550404). Requirements for use of these data sets are listed on the synapse pages for each data set. HBTRC data was downloaded from the Gene Expression Omnibus (GEO) under accession code GSE44772.  Human blood gene expression and phenotype data from the AddNeuroMed cohort are available from GEO under accession codes GSE63060 and GSE63061. Mouse brain gene expression samples and associated phenotypes are available from GEO under accession code GSE64398.

**Pathways and gene sets:** In our analyses, we evaluated our results with respect to publically available gene sets. These include REACTOME and KEGG pathways available from MSigDB (c2 pathways v7.0). We also obtained gene signatures from [Olah et al. (2020)](https://doi.org/10.1038/s41467-020-19737-2) and [Mathys et al. (2019)](https://doi.org/10.1038/s41586-019-1195-2) (each available as supplementary data from the respective papers).


