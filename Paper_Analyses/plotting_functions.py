import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches





def get_colors_legends(labels, colormapname, legend_alpha):
    
    # set colormap
    mymap = plt.get_cmap(colormapname)
    
    # create new labels array -- integers from 0 to the max number of labels
    # legend array -- will store patches for the legend
    # colors -- store a list of colors, one for every point from the labels
    newlabs = np.empty(len(labels))
    legend = []
    colors = np.ones([len(labels), 4])    

    
    # iterate through unique label values --> add colors for datapoints that match those values
    # add a patch for the legend
    for i, lab in enumerate(np.unique(labels[labels.astype(str) != "nan"])):
        
        cur_idx = np.where(labels == lab)[0]
        newlabs[cur_idx] = i
        legend.append(mpatches.Patch(color=mymap(i), label=lab, alpha=legend_alpha))
        colors[cur_idx] = np.array(mymap(i))
        
    return colors, legend



def color_mapper(cmap, minima, maxima):
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

def pvals_to_colors(nlog10_pval_array, cmap, significance_cutoff, minimum, maximum):   
    color_list = []
    for elt in nlog10_pval_array:
        if elt < -np.log10(significance_cutoff):
            color_list.append([1,1,1,1])
        else:
            color_list.append(color_mapper(cmap, minimum, maximum).to_rgba(elt))
    return np.array(color_list)



def vals_to_colors(array, cmap, minimum, maximum, nangray = False):   
    
    color_list = []
    for elt in array:
        if np.isnan(elt):
            #if nangray flag is set to true, draw a gray dot 
            if nangray:
                color_list.append([.5,.5,.5,1])

            else:
                color_list.append([0,0,0,1])
        else:
            color_list.append(color_mapper(cmap, minimum, maximum).to_rgba(elt))
    return np.array(color_list)

