import numpy as np
import scipy.io as sio
import os


def hyp_epoching(hyp=[],epoch_length=30):
    m=1
    n=1
    hyp_norm=np.array([])
    while m<len(hyp-epoch_length):
        hyp_norm=np.hstack((hyp_norm,hyp[m])) if hyp_norm.size else hyp[m]
        n+=1
        m+=30
    return hyp_norm
def assign_sleep_stage(data,hyp):

    hyp[hyp==3]=4
    AWA=data[np.where(hyp==0)[0]]
    N1=data[np.where(hyp==1)[0]]
    N2=data[np.where(hyp==2)[0]]
    N3=data[np.where(hyp==4)[0]]
    Rem=data[np.where(hyp==5)[0]]
    stages={'AWA':AWA,'N1':N1,'N2':N2,'N3':N3,'Rem':Rem}

    return stages
