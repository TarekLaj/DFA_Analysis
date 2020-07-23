import scipy.io as sio
import numpy as np
from DFA_utils import (CompHurst,
                      compRMS,
                      get_data,
                      data_epoching,
                      get_amplitude,
                      MDFA1,
                      like_rwalk)
import os


#-------------------------------------------------------------------------------
#Load data
#-------------------------------------------------------------------------------
fs=1000
data_path='/media/ronin_cunningham/StorageDevice/Data sleep full night mat/'
#data_path='/home/karim/projet_c1_c2_dreamer'
scales=[64,128,256,512,1024,2048,4096]
qs=[-5,-3,-1,0,1,3,5]
m=1
filename=os.path.join(data_path,'s{}_sleep.mat')
save_path='/media/karim/DATAS/DFA_Analysis/DFA_values/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
sbj_list = np.array([k for k in range(1, 38)])
results=dict()
for sbj in sbj_list:
    H_all,Fq_all,Hq_all,hq_all,tq_all,Dq_all=[],[],[],[],[],[]
    m_data=get_data(filename.format(str(sbj)))
    X=data_epoching(data=m_data,epoch_length=30,Fs=fs)
    del m_data
    # print(X.shape)
    # (19, 30000, 947)
    for ep in range(X.shape[2]):
        print('Subject{s} epoch {e} '.format(s=str(sbj),e=str(ep)))
        H_elect,Fq_elect,Hq_elect=np.array([]),np.array([]),np.array([])
        hq_elect,tq_elect,Dq_elect=np.array([]),np.array([]),np.array([])
        for elect in range(X.shape[0]):

            H,reg=CompHurst(data=np.squeeze(X[elect,:,ep]),scales=scales,m=1)
            if H<=0.2:
                data=like_rwalk(signal=np.squeeze(X[elect,:,ep]),fs=fs,plot=False)
                print('Random like transformation...')
            elif (H>0.2 and H<=1.2):
                data=np.squeeze(X[elect,:,ep])
                print('Random like signal...')
            elif H>1.2:
                print('Computing derivate...')
                data=np.diff(np.squeeze(X[elect,:,ep]))
            print('Computing Hq')
            Fq,Hq,hq,tq,Dq=MDFA1(data=data,scales=scales,qs=qs,m=m)
            H_elect=np.vstack((H_elect,H)) if H_elect.size else H
            Fq_elect=np.vstack((Fq_elect,Fq)) if Fq_elect.size else Fq
            Hq_elect==np.vstack((Hq_elect,Hq)) if Hq_elect.size else Hq

            hq_elect=np.vstack((hq_elect,hq)) if hq_elect.size else hq
            tq_elect=np.vstack((tq_elect,tq)) if tq_elect.size else tq
            Dq_elect=np.vstack((Dq_elect,Dq)) if Dq_elect.size else Dq

        H_all.append(H_elect)
        Fq_all.append(Fq_elect)
        Hq_all.append(Hq_elect)
        hq_all.append(hq_elect)
        tq_all.append(tq_elect)
        Dq_all.append(Dq_elect)
    results={'Hurst':H_all,
            'Fq':Fq_all,
            'Hq':Hq_all,
            'hq':hq_all,
            'tq':tq_all,
            'Dq':Dq_all}
    sio.savemat(os.path.join(save_path,'mfdfa_s{}_diff.mat'.format(str(sbj))),results)
