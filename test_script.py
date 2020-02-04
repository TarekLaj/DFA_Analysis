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
data_path='/home/karim/DFA_Analysis/'
#data_path='/home/karim/projet_c1_c2_dreamer'
file_to_load=os.path.join(data_path,'test_data.npy')
save_path='/home/karim/DFA_Analysis/DFA_values/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

#X=np.load(file_to_load)
X=sio.loadmat('/home/karim/DFA_Analysis/packages/fractaldata.mat')['multifractal']
#data=np.cumsum(X-np.mean(X)).T

scales=[16,32,64,128,256,512,1024]
qs=[-5,-3,-1,0,1,3,5]
#amp=get_amplitude(signal=data,fs=1000,plot=False)
# Step1: convert signal to like random walk signal
#Rw_signal=like_random_walk(signal=amp,fs=1000,plot=False)
# Step 2: compute beta DFA
# scales=[2*fs,5*fs,10*fs, 20*fs, 30*fs]
# qs=[-2, -1,0,1 ,2]
m=1

data=like_rwalk(signal=X,fs=fs,plot=False)
H,reg=CompHurst(data=data,scales=scales,m=1)
Fq,Hq,hq,tq,Dq=MDFA1(data=data,scales=scales,qs=qs,m=m)
print('Hurst',H)
print('Fq',Fq)
print('Hq',Hq)
print('hq',hq)
print('tq',tq)
print('Dq',Dq)
