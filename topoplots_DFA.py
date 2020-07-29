import numpy as np
import scipy.io as sio
import os
from classification_utils import Topo_DA
from plot_utils import get_data


#load data and extract hurst, mfdfa values
main_path='/home/karim/DFA_Analysis/'
data_path=os.path.join(main_path,'DFA_values_intgr')
sensors_pos = sio.loadmat('/home/karim/projet_c1_c2_dreamer/Coord_2D_Slp_EEG.mat')['Cor']

sDreamer = np.arange(18) +1
sNnDreamer = np.array([k for k in range(19, 37)])

stages=['AWA','N1','N2','N3','Rem']
data_to_plot='MFDFA'
savepath = '/home/karim/results_C1C2/figures_finales/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

for i_st,st in enumerate(stages):
    
    Drdata=get_data(path=data_path,
                    subj_list=sDreamer,
                    data_to_load=data_to_plot,
                    stage='AWA',
                    moy=1)

    nnDrdata=get_data(path=data_path,
                      subj_list=sNnDreamer,
                      data_to_select='MFDFA',
                      stage='AWA',
                      moy=1)
