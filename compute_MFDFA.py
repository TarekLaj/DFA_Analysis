import scipy.io as sio
import numpy as np
import h5py
from DFA_utils import get_data,data_epoching,get_amplitude,MDFA
import os

#-------------------------------------------------------------------------------
#Load data
#-------------------------------------------------------------------------------
fs=1000
data_path='/media/ronin_cunningham/StorageDevice/Data sleep full night mat/'
#data_path='/home/karim/projet_c1_c2_dreamer'
file_to_load=os.path.join(data_path,'s{}_sleep.mat')
save_path='/home/karim/DFA_Analysis/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
sbj_list = np.array([k for k in range(1, 38)])
for sbj in sbj_list:
    data=get_data(file_to_load.format(str(sbj)))
    print(data.shape)
    segments=data_epoching(data=data,epoch_length=30,Fs=fs)
    print(segments.shape)
    del data
    n_elect,time,n_seg=segments.shape
    print(n_elect,time,n_seg)

    #signal=np.reshape(segments[1,:,1],(1,time))
    #print(signal.shape)
    # step0 : compute amplitude using hilbert transform
    amp=get_amplitude(signal=segments[1,:,1],fs=1000,plot=False)
    print(save_path+'test_data.npy')
    np.save(save_path+'test_data.npy',amp)
    # Step1: convert signal to like random walk signal
    #Rw_signal=like_random_walk(signal=amp,fs=1000,plot=False)
    # Step 2: compute beta DFA
    scales=[2*fs,5*fs,10*fs, 20*fs, 30*fs]
    qs=[-2, -1, 1 ,2]
    Fq,Hq,hq,tq,Dq=MDFA(amp,scales,qs)
    print(Fq,Hq,hq,tq,Dq)
    1/0
    # Step 3: compute Md MFDFA
