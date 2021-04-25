import numpy as np
import pandas as pd
import matplotlib
import re
from matplotlib import pyplot as plt

"""
~~~~~~~~~~~~~~~~~~~~~
       INPUTS
~~~~~~~~~~~~~~~~~~~~~

- RANKED_FEATURES:   Pandas dataframe of features by name/rankings. One of the columns should be titled "feature" and the values in the column should be the names of your features. The other columns should provide rankings for your data in the form of percentiles (ranging between 0 and 1)

- CATEGORIES: A list of categories you want to generate plots for (if you just want one category, enter it as "[category_name]")

- CATEGORY_FEATURES: A dictionary:  {category_name: np.array([list of features belonging to that category])}
            ### (The items in CATEGORIES should all have keys in the dictionary)

- LEFT_ and RIGHT_METHOD_NAME:   Column names in RANKED_FEATURES that you want to compare

- MAX_COLS = optional, # of columns you want your subplots to have (helpful if you want to make plots for many categories)

- color_lim = percentile of features you want to plot (e.g., color_lim=.99 means you'll generate colored blue/red/purple lines for the top 1% of features from either of the two methods you're checking) 

- display_lim = should be less than or equal to color_lim. This parameter allows you to plot additional lines that don't meet the color threshold but are still somewhat highly ranked. For example if you do display_lim=.95 and color_lim=.99, you'll draw red/blue/purple lines for features ranked in the top 1% of either ranking, but will draw grey lines for features ranked in the top 5% of either ranking

"""

def rank_comparison_plot(RANKED_FEATURES, CATEGORIES, CATEGORY_FEATURES, LEFT_METHOD_NAME, RIGHT_METHOD_NAME, MAX_COLS=4, display_lim=.99, color_lim=.99):
    # loading color palette for paper
    cmap=plt.cm.tab20
    cmaplist = [cmap(i) for i in range(cmap.N)]
    missing_color="#eeeeee"
    print("[red, blue, purple, grey]")

    numcols = min(len(CATEGORIES),MAX_COLS)
    numrows = int(np.ceil(len(CATEGORIES)/numcols))
    fs = (numcols*5, numrows*5)

    if (numrows>1) and (numcols>1):
        f,axs = plt.subplots(numrows, numcols, figsize=fs)
    elif numcols > 1:
        f,axs = plt.subplots(1, numcols, figsize=fs)
    elif numrows > 1:
        f,axs = plt.subplots(numrows, 1, figsize =fs)
    else:
        f,ax = plt.subplots(figsize=fs)

    # return # of features that satisfy the color limit for at least one side, along with the ratios calculated
    feature_counts = []
    ratios = []
    
    for catnum, cat in enumerate(CATEGORIES):
        # get ax
        if numrows > 1 and numcols > 1:
            r=int(catnum/numcols)
            c=int(catnum%numcols)
            ax = axs[r,c]
        elif numrows > 1 or numcols > 1:
            ax = axs[catnum]

        # get rankings for features in the category
        features_to_plot = CATEGORY_FEATURES[cat]
        features_ranking_comparison = pd.DataFrame(features_to_plot, columns=["feature"]).merge(RANKED_FEATURES, on="feature", how="inner")

        # Add some grid-lines for clarity 
        ax.hlines(0, 0,1, alpha=.5)
        ax.hlines(.5, 0,1, alpha=.5)
        ax.hlines(.75, 0,1, alpha=.5)
        ax.hlines(.9, 0,1, alpha=.5)
        ax.hlines(1,0,1, alpha=.5)

        # Plot colored lines
        selected_colors = []
        for i,row in features_ranking_comparison.iterrows():
            val_l = row[LEFT_METHOD_NAME]
            val_r = row[RIGHT_METHOD_NAME]
            if (val_l > display_lim) or (val_r > display_lim):
                if abs(val_l)>color_lim and abs(val_r)>color_lim:
                    sel_color = cmaplist[8] # purple (tie - both left and right are significant)
                elif abs(val_r)>color_lim:
                    sel_color = cmaplist[0] # blue (right side significant, left side not)
                elif abs(val_l)>color_lim:
                    sel_color = cmaplist[6] # red (left side significant, right side not)
                else:
                    sel_color = "#aaaaaa"                
                selected_colors.append(matplotlib.colors.to_hex(sel_color))
                ax.plot([0.02,.98], [val_l, val_r],alpha=.5, c=sel_color)
        

        # Count how many lines we have of each color so proportions can be put in title
        color_names = ["red", "blue", "purple", "gray"]
        color_hexs = ['#d62728', '#1f77b4', '#9467bd', "#aaaaaa"]
        color_counts=[]
        for i,c in enumerate(color_names):
            color_counts.append(len(np.where(np.array(selected_colors) == color_hexs[i])[0]))
        if (sum(color_counts[:3])>1):
            feature_count = sum(color_counts[:3])
            numerator = (color_counts[0]+color_counts[2])
            denominator = (color_counts[1]+color_counts[2])
            if numerator==0 or denominator==0:
                ratio = np.nan
            else:
                ratio = numerator/denominator
        else:
            feature_count = 0
            ratio = 1
        print(cat, color_counts)

        feature_counts.append(feature_count)
        ratios.append(ratio)
    
    
        # Title: category name,  # features in the top x of either ranking system,  ratio of left to right high rankings
        title = "%s\n%i Features;  Left/Right Ratio: %.2f"%(cat, feature_count, ratio)

        # Formatting
        ax.set_facecolor('None')
        ax.set_title(title)    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axis('off')
        plt.gca().axes.get_yaxis().set_visible(False)


    plt.tight_layout(h_pad=6, w_pad=0)
    
    if numrows>1 or numcols>1:
        return np.array(feature_counts), np.array(ratios), axs
    else:
        return np.array(feature_counts), np.array(ratios), ax
    
    

#### GET dicts for gene sets to categories 
def get_geneset_cats(gset, path_to_gs_data, gmt_locations):
    
    def reverse_dict(dic):
        newdict = {}
        for val in np.unique(list(dic.values())):
            newdict[val] = np.array([k for k,v in dic.items() if v == val])
        return newdict

    gmt = pd.read_csv(gmt_locations[gset], header=None)
    gmt["names"] = gmt[0].apply(lambda x: x.split("\t")[0])
    gmt["genes"] = gmt[0].apply(lambda x: x.split("\t")[2:])


    if gset == "REACTOME":
        gset_level = 0

        HIERARCHY = pd.read_csv(path_to_gs_data + "pathway_hierarchies/REACTOME/HIERARCHY.txt", delimiter="\t", index_col=0)

        KEEP_CATEGORIES = ["REACTOME_AUTOPHAGY", 'REACTOME_CELL_CYCLE', 'REACTOME_CELL_CELL_COMMUNICATION', 'REACTOME_CELLULAR_RESPONSES_TO_EXTERNAL_STIMULI',
        "REACTOME_CHROMATIN_ORGANIZATION", "REACTOME_CIRCADIAN_CLOCK", "REACTOME_DEVELOPMENTAL_BIOLOGY", "REACTOME_DIGESTION_AND_ABSORPTION",
        "REACTOME_DISEASE", "REACTOME_DNA_REPAIR", "REACTOME_DNA_REPLICATION", "REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION", "REACTOME_GENE_EXPRESSION_TRANSCRIPTION",
        "REACTOME_HEMOSTASIS", "Immune System", "Metabolism", 'Metabolism of proteins', 'REACTOME_METABOLISM_OF_RNA', "REACTOME_MUSCLE_CONTRACTION",
        "REACTOME_NEURONAL_SYSTEM", "REACTOME_ORGANELLE_BIOGENESIS_AND_MAINTENANCE", "REACTOME_PROGRAMMED_CELL_DEATH", "REACTOME_PROTEIN_LOCALIZATION",
        "REACTOME_REPRODUCTION", "Signal Transduction", "REACTOME_TRANSPORT_OF_SMALL_MOLECULES", "REACTOME_VESICLE_MEDIATED_TRANSPORT"]

        def query_hierarchy(df, pathway, level):
            rows,cols = np.where(df.isin([pathway]))
            if len(rows)==0:
                return "UNKNOWN"
            else:
                ## sometimes, a single gene set maps to multiple categories. Choose the most common one from the list of approved categories
                all_cats_counts =  np.unique(HIERARCHY.iloc[rows,level],return_counts=True)
                filtered = [x for x in all_cats_counts[0] if x in KEEP_CATEGORIES]
                if len(filtered) == 0:
                    return "UNKNOWN"
                else:
                    return filtered[0]

        geneset_to_category = {}
        for pw in gmt["names"]:
            geneset_to_category[pw] = query_hierarchy(HIERARCHY, pw, gset_level)            

    elif gset == "KEGG":
        gset_level="MEDIUM"

        HIERARCHY = pd.read_csv(path_to_gs_data + "pathway_hierarchies/KEGG/HIERARCHY.txt", delimiter="\t", names=["TOP", "MEDIUM", "PWAY"])

        def query_hierarchy(KEGG_HIERARCHY, pway, level):
            return KEGG_HIERARCHY[KEGG_HIERARCHY["PWAY"]==pway][gset_level].values

        geneset_to_category = dict(zip(HIERARCHY["PWAY"], HIERARCHY[gset_level]))

    elif gset == "ben_modules":
        ben_clusters_to_consensus = pd.read_csv(path_to_gs_data + "ben_modules/cluster_to_consensus.csv")
        geneset_to_category = dict(zip(ben_clusters_to_consensus["Module Name"], ben_clusters_to_consensus["Consensus Cluster"]))


    elif gset=="mathys_celltypes":
        geneset_to_category = {}
        for module in gmt["names"]:
            geneset_to_category[module] = re.sub(r'\d+$', '', module)

    elif gset=="sara_modules":
        sara_dict = {"m7": "tau", "m109": "dementia", "m110":"abeta", "m116":"age"}
        geneset_to_category = {}
        for module in gmt["names"]:
            if module in sara_dict.keys():
                geneset_to_category[module] = sara_dict[module]
            else:
                geneset_to_category[module] = "Uncategorized"

    else:
        # NO CATEGORIES
        gmt = pd.read_csv(gmt_locations[gset], header=None)
        gmt["names"] = gmt[0].apply(lambda x: x.split("\t")[0])
        gmt["genes"] = gmt[0].apply(lambda x: x.split("\t")[2:])

        geneset_to_category = {}
        for elt in gmt["names"]:
            geneset_to_category[elt] = elt

    category_to_geneset = reverse_dict(geneset_to_category)
    return gmt, category_to_geneset, geneset_to_category
    