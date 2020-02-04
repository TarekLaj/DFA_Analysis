
import scipy.io as sio
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
def get_data(filename):
    with h5py.File(filename, 'r') as f:
        contents= f['m_data'][:]
    return contents[:,0:19]

def data_epoching(data=[],epoch_length=30,Fs=1000):
    print('Epoching data into {} sec segments...' .format(str(epoch_length)) )
    if data.shape[0]>data.shape[1]:
        print('warning data shape must be elect*time...')
        print('reshaping data')
        data=data.T
    n=epoch_length*Fs
    L=data.shape[1]
    a_extrat=int((-L)%n)
    x_temp=np.concatenate((data,np.zeros((data.shape[0],a_extrat))),axis=1)
    nbre_epochs=int(x_temp.shape[1]/n)

    X=np.split(x_temp,nbre_epochs,axis=1)
    return np.dstack(X)

def get_amplitude(signal=[],fs=1000,plot=False):
    from scipy.signal import hilbert
    amp_signal = hilbert(signal)
    amplitude  = np.abs(amp_signal)
    if plot:
        time=np.linspace(0, signal.shape[0]/fs, signal.shape[0], endpoint=False)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time, signal, label='signal')
        ax.plot(time, amplitude, label='envelope')
        plt.show()
        return amplitude,ax
    else:
        return amplitude
def like_rwalk(signal=[],fs=1000,plot=False):
    randw=np.cumsum(signal-np.mean(signal))
    if plot:
        time=np.linspace(0, signal.shape[0]/fs, signal.shape[0], endpoint=False)
        A = 100
        fig = plt.figure()
        plt.plot(time,A*signal)
        plt.plot(time,randw,'r',lw=1.5)
        plt.ylabel('signal\namplitude',ha='center')
        plt.show()
        return randw,fig
    else:
        return randw

def compRMS(X=[],scales=[],m=1):
    t = np.arange(X.shape[0])
    step = scales[0]
    i0s = np.arange(0,X.shape[0],step)
    RMS = np.zeros((len(scales),i0s.shape[0]),'f8')
    for si,scale in enumerate(scales):
        s2 = scale//2
        for j,i0 in enumerate(i0s-s2):
            i1 = i0 + scale
            if i0 < 0 or i1 >= X.shape[0]:
                RMS[si,j] = float('nan')
                continue
            t0 = t[i0:i1]
            C = np.polyfit(t0,X[i0:i1],m)
            fit = np.polyval(C,t0)
            RMS[si,j] = np.sqrt(((X[i0:i1]-fit)**2).mean())
    return RMS


def CompHurst(data=[],scales=[],m=1):

    rRMS,F,Idx_sc_all=[],[],[]
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
        F.append(np.sqrt(np.square(rms).mean())) #over all hurst

    Cx=np.polyfit(np.log2(scales),np.log2(F),m)
    H=Cx[0]
    Regline=np.polyval(Cx,np.log2(scales))
    return H,Regline
def ComputeFq(data=[],scales=[],qs=[],m=1):
    rRMS,Idx_sc_all=[],[]
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

        Fq[np.asarray(qs)==0,si]=np.exp(0.5*(np.log(rRMS[si]**2.0)).mean(0))
    return Fq
def MDFA1(data=[],scales=[],qs=[],m=1):
    Fq=ComputeFq(data=data,scales=scales,qs=qs,m=1)
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

    return Fq, Hq, hq, tq, Dq
