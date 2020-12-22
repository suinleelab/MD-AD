import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
import plotting_functions


def plot_categorical_tsne(E, phens, phen, cmap, nangray, alpha=.4, save_path=None):

    fig, axes = plt.subplots(1,1,figsize=(4, 4))

    fig.suptitle(phen)
    c, legend_patches = plotting_functions.get_colors_legends(phens, cmap, 1)
    c[:,3]=alpha
    nan_idx = np.where(pd.Series(phens).isnull())

    if len(nan_idx[0])>0:
        if nangray:
            c[np.where(pd.Series(phens).isnull())]=np.array([.5,.5,.5,.05])

        else:
            c[np.where(pd.Series(phens).isnull())]=np.array([0,0,0,0])

    edge_colors = np.zeros([len(c),4])                    
    if nangray:
        edge_colors[np.where(pd.Series(phens).isnull())] = np.array([.5,.5,.5,.2])

    axes.scatter(E.T[0],E.T[1], c=c, edgecolors=edge_colors)
    axes.legend(handles=legend_patches, bbox_to_anchor=(.5, -.25), ncol =len(legend_patches), loc='lower center');
    axes.axis('off')
    axes.set_facecolor('None')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + "%s.png"%phen, bbox_inches="tight")
   
     
        
def plot_continuous_tsne(E, phens, phen, cmap, nangray, lims=None, save_path=None):
    phens = phens.astype(float)


    fig, axes = plt.subplots(2,1,figsize=(4, 5), gridspec_kw = {'height_ratios':[10, 1]})

    #print(phens)
    if lims==None:
        c = plotting_functions.vals_to_colors(phens, cmap, np.nanmin(phens),np.nanmax(phens))
    else:
        c = plotting_functions.vals_to_colors(phens, cmap, lims[0], lims[1])


    c[:,3]=.4
    if nangray:
        c[np.where(np.isnan(phens))]=np.array([.5,.5,.5,.4])
        c[np.where(np.isnan(phens))]=np.array([.5,.5,.5,.05])

    else:
        c[np.where(np.isnan(phens))]=np.array([0,0,0,0])


    edge_colors = np.zeros([len(c),4])                    
    if nangray:
        edge_colors[np.where(pd.Series(phens).isnull())] = np.array([.5,.5,.5,.2])


    fig.suptitle(phen)
    # T-SNE PLOT
    axes[0].scatter(E.T[0],E.T[1], c=c, edgecolors=edge_colors)

    # SHOW COLORBAR
    if lims==None:
        norm = mpl.colors.Normalize(vmin=np.nanmin(phens), vmax=np.nanmax(phens))
    else:
        norm = mpl.colors.Normalize(vmin=lims[0], vmax=lims[1])
    
    
    cb1 = mpl.colorbar.ColorbarBase(axes[1], cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('value')
    
    axes[0].axis('off')
    axes[0].set_facecolor('None')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig.savefig(save_path + "%s.png"%phen)