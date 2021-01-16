import h5py 
import numpy as np
import sys
import pickle
import os
from sklearn.cluster import KMeans
import pandas as pd
from scipy import stats 


from configs import * 


clusternum = 50
reps = np.arange(100)
HIDDEN_LAYER = 1



######### LOAD EMBEDDINGS FROM ALL RUNS ###############################################################

print("Loading embeddings...")

X_transformed_dicts = {}

if HIDDEN_LAYER in [0,1]:
    for rep in reps:
        X_transformed = np.loadtxt("%sMTL/%i/%i.txt"%(final_rep_embeddings_savepath, HIDDEN_LAYER, rep))
        X_transformed_dicts[rep] = X_transformed
        
if HIDDEN_LAYER in [2,3]:
    for phenotype in phenotypes:
        if phenotype not in X_transformed_dicts.keys():
            X_transformed_dicts[phen] = {}

        for rep_name in  os.path.isdir("%sMTL/%i/%s/"%(final_rep_embeddings_savepath, HIDDEN_LAYER, phenotype)):
            rep = int(rep_name.split(".")[0])
            X_transformed = np.savetxt("%sMTL/%i/%s/%i.txt"%(final_rep_embeddings_savepath, HIDDEN_LAYER, phenotype, rep))
            X_transformed_dicts[phen][rep] = X_transformed



######### COMBINE RUNS & NORMALIZE NODE ACTIVATIONS ###################################################
print("Combining runs...")


non_triv_idxes = []
runs = []
transformed_vals_prestacked = []
for rep in reps:
    non_triv_idx = np.where(np.mean(X_transformed_dicts[rep] != 0, axis=0)>0)[0]
    non_triv_idxes.append(non_triv_idx)
    runs.append(np.array([rep]*len(non_triv_idx)))
    transformed_vals_prestacked.append(X_transformed_dicts[rep].T[non_triv_idx])
    
    
transformed_vals_stacked = np.vstack(transformed_vals_prestacked)
runs = np.hstack(runs)
transformed_vals_stacked_zscore = stats.zscore(transformed_vals_stacked,axis=1)

print("Average # of kept nodes:", np.mean([len(x) for x in non_triv_idxes]))


######### RUN K-MEANS CLUSTERING OVER ALL RUNS (CLUSTER NODES BY SIMILARITY ACROSS SAMPLES) ##########
print("Clustering nodes...")

c = KMeans(n_clusters=clusternum).fit(transformed_vals_stacked_zscore)
cluster_df = pd.DataFrame(np.vstack([runs, np.hstack(non_triv_idxes), c.labels_]).T, columns=["run", "node_idx", "cluster"])


########## IDENTIFY NODEs CLOSEST TO THE CENTERS OF THEIR CLUSTERS ###################################

def cluster_medoid(zscores_for_cluster):
    distMatrix=np.array([[np.dot(x-y,x-y) for y in zscores_for_cluster] for x in zscores_for_cluster])
    idx = np.argmin(distMatrix.sum(axis=0))
    return idx

selected_medoids = []
medoid_activations = []
for cluster_id in range(clusternum):
    rows = np.where(cluster_df["cluster"]==cluster_id)[0]
    selected_idx = cluster_medoid(transformed_vals_stacked_zscore[rows])
    medoid_activations.append(transformed_vals_stacked_zscore[selected_idx])
    selected_medoids.append(rows[selected_idx])
    
new_embedding = np.array(medoid_activations).T


########### SAVE CENTROIDS + INFO LINKING THEM TO ORIGINAL RUNS ######################################

s="%s%i/normed_KMeans_medoids/MTL_%i.p"%(final_rep_consensus_embeddings_savepath, HIDDEN_LAYER,clusternum)
os.makedirs(os.path.dirname(s), exist_ok=True)
with open(s,"wb") as f:
    pickle.dump([new_embedding, cluster_df, c], f)
print("Saved consensus embedding to", s)
# save selected medoids
s="%s%i/normed_KMeans_medoids/MTL_%i_medoids_info.csv"%(final_rep_consensus_embeddings_savepath, HIDDEN_LAYER, clusternum)
cluster_df.iloc[selected_medoids].to_csv(s, index=False)
print("Saved sources of centroids to", s)
