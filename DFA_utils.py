
import scipy.io as sio
import numpy as np
import h5py
import matplotlib.pyplot as plt

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


def compRMS(X=[],scales=[],m=1,verbose=False):

    t = np.arange(X.shape[0])
    step = scales[0]
    i0s = np.arange(0,X.shape[0],step)
    RMS = np.zeros((len(scales),i0s.shape[0]),'f8')
    for si,scale in enumerate(scales):
        if verbose: print ('.')
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
def compFq(rms,qs):
    """Compute scaling function F as:

      F[scale] = pow(mean(RMS[scale]^q),1.0/q)
    This function computes F for all qs at each scale.
    The result is a 2d NxM array (N = rms.shape[0], M = len(qs))
    Parameters
    ----------
    rms:    the RMS 2d array (RMS for scales in rows) computer by compRMS or fastRMS
    qs:     an array of q coefficients
    """
    Fq = np.zeros((rms.shape[0],len(qs)),'f8')
    mRMS = np.ma.array(rms,mask=np.isnan(rms))
    for qi in range(len(qs)):
        p = qs[qi]
        Fq[:,qi] = (mRMS**p).mean(1)**(1.0/p)
    
    Fq[:,qs==0] = np.exp(0.5*(np.log(mRMS**2.0)).mean(1))[:,None]
    return Fq



def MDFA(X,scales,qs):
        RW = like_rwalk(X)
        RMS = compRMS(RW,scales)
        Fq = compFq(RMS,qs)
        Hq = np.zeros(len(qs),'f8')
        for qi,q in enumerate(qs):
            C = polyfit(log2(scales),log2(Fq[:,qi]),1)
            Hq[qi] = C[0]
            if abs(q - int(q)) > 0.1: continue
            loglog(scales,2**np.polyval(C,np.log2(scales)),lw=0.5,label='q=%d [H=%0.2f]'%(q,Hq[qi]))
        tq = Hq*qs - 1
        hq = np.diff(tq)/(qs[1]-qs[0])
        Dq = (qs[:-1]*hq) - tq[:-1]
        return Fq, Hq, hq, tq, Dq
