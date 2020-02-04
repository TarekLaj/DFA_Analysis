#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:32:40 2020

@author: karim
"""


import numpy as np
import math
import scipy.io as sio
import os
from DFA_utils import like_rwalk
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

X=sio.loadmat('/home/karim/DFA_Analysis/packages/fractaldata.mat')['multifractal']
scales=[16,32,64,128,256,512,1024]
qs=[-5,-3,-1,0,1,3,5]
m=1


data=like_rwalk(signal=X,fs=fs,plot=False)
rRMS,F,Idx_sc_all=[],[],[]

Fq = np.zeros((len(scales),len(qs)),'f8')
for si,sc in enumerate(scales):
    segments=math.floor(data.shape[0]/sc)
    rms=[]
    for v in range(segments):
        idx_start=v*sc
        idx_stop=v*sc+sc
        idx_sc=np.arange(idx_start,idx_stop)
        Idx_sc_all.append(idx_sc)
        X_idx=data[idx_sc]
        C=np.polyfit(idx_sc,X_idx,m)
        fit=np.polyval(C,idx_sc)
        sq=np.square(X_idx-fit).mean(axis=0)
        rms.append(np.sqrt(sq))
    rRMS.append(np.array(rms))

    for nqi,nq in enumerate(qs):

        if nq!=0:

            Fq[nqi,si] =(rRMS[si]**nq).mean()**(1/nq)
        else:
            Fq[nqi,si]=np.nan



    #qs=np.asarray(qs)
#
#    idx=np.where(qs==0)[0][0]
#    Fq[qs==0,si]=np.exp(0.5*(np.log(rRMS[si]**2.0)).mean(0))
    Fq[np.asarray(qs)==0,si]=np.exp(0.5*(np.log(rRMS[si]**2.0)).mean(0))
Hq=[]
for qi,q in enumerate(qs):
    C1 = np.polyfit(np.log2(scales),np.log2(Fq[qi,:]),1)
    Hq.append(C1[0])
    if abs(q - int(q)) > 0.1: continue
qs=np.asarray(qs)
Hq=np.asarray(Hq)

tq = Hq*qs - 1
hq = np.diff(tq)/(qs[1]-qs[0])
Dq = (qs[:-1]*hq) - tq[:-1]
