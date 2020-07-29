import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mne.viz import plot_topomap
from mne import pick_types, find_layout
import seaborn as snb
import os
import pickle as pkl


def get_data(path=[],subj_list=[],data_to_load=[],stage=None,moy=0):
    Grp_data=np.array([])
    for ids,s in enumerate(subj_list):
        data_file = open(os.path.join(path,'{d}_stages_s{s}.pkl'.format(d=data_to_select,s=int(s))), "rb")
        data = pkl.load(data_file)
        x=data.get(stage)
        x_data=np.delete(x,np.unique(np.argwhere(np.isnan(x))[:,0]),0)

        if moy==1:
            x_data=np.mean(x_data,axis=0)
        Grp_data=np.vstack((Grp_data,x_data)) if Grp_data.size else x_data

    return Grp_data

def Topo_DA(DA=[],sensors_pos=[],mask=False,DA_thr=None,save_file=None):
    if mask:
        mask_default = np.full((len(DA)), False, dtype=bool)
        mask = np.array(mask_default)
        mask[DA >= DA_thr] = True
        mask_params = dict(marker='*', markerfacecolor='w', markersize=18) # significant sensors appearence
        fig = plt.figure(figsize = (10,5))
        ax,_ = plot_topomap(DA,sensors_pos,
                            cmap='viridis',
                            show=False,extrapolate='local',
                            vmin=50,vmax=70,
                            contours=True,
                            mask = mask,
                            mask_params = mask_params)

        #fig.colorbar(ax, shrink=0.25)
        if save_file:
            plt.savefig(save_file, dpi = 300)

    else:
        fig = plt.figure(figsize = (10,5))
        ax,_ = plot_topomap(DA, sensors_pos,cmap='viridis',show=False,
        vmin=50,vmax=70,contours=True)

        #fig.colorbar(ax, shrink=0.25)
        if save_file:
            plt.savefig(save_file, dpi = 300)
    return ax
