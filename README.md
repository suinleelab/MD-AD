# MD-AD

Code for paper "Unified AI framework to uncover deep interrelationships between gene expression and Alzheimer’s disease neuropathologies". Please read our preprint at the following link: https://doi.org/10.1101/2020.11.30.404087

### Data Availability:
**Gene expression and phenotype data:** The results published here are in based on data obtained from the [AD Knowledge Portal](https://adknowledgeportal.synapse.org/). Postmortem brain gene expression samples and phenotype labels are available from the AD Knowledge Portal for the following data sets (with listed synapse.org Synapse IDs):  ACT (syn5759376), ROSMAP (syn3219045), MSBB (RNA Sequencing: syn3159438, Microarray: syn3157699), Mayo Clinic Brain Bank (syn5550404). Requirements for use of these data sets are listed on the synapse pages for each data set. HBTRC data was downloaded from the Gene Expression Omnibus (GEO) under accession code GSE44772.  Human blood gene expression and phenotype data from the AddNeuroMed cohort are available from GEO under accession codes GSE63060 and GSE63061. Mouse brain gene expression samples and associated phenotypes are available from GEO under accession code GSE64398.

**Pathways and gene sets:** In our analyses, we evaluated our results with respect to publically available gene sets. These include REACTOME and KEGG pathways available from MSigDB (c2 pathways v7.0). We also obtained gene signatures from [Olah et al. (2020)](https://doi.org/10.1038/s41467-020-19737-2) and [Mathys et al. (2019)](https://doi.org/10.1038/s41586-019-1195-2) (each available as supplementary data from the respective papers).


#### Code to train models:
- Training MD-AD
- Training MLP / Linear baselines
- Extracting embeddings
- Extracting gene importances with Integrated Gradients

#### Code to perform analyses and generate figures:

- Figure 2a - Paper_Analyses/CV_Prediction_Performance.ipynb
- Figure 2b - Paper_Analyses/Subsets_CV_ROSMAP_Plots.ipynb
- Figures 2c-d, 7a-b - Paper_Analyses/External_validation_predictions.ipynb
- Figures 3a-c - Paper_Analyses/Evaluate_Embedding_Correlations.ipynb and Paper_Analyses/t-SNE_plots_internal.ipynb
- Figures 3d-e, 7c - Paper_Analyses/t-SNE_embeddings_external.ipynb
- Figure 4a-b, 6c - 
- Figure 4c - 
- Figures 5 and 6a-b - 

