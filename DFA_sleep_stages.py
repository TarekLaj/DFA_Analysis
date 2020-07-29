import numpy as np
import scipy.io as sio
import os
import pickle
from preproc_utils import assign_sleep_stage,hyp_epoching


main_path='/home/karim/DFA_Analysis/'
data_path=os.path.join(main_path,'DFA_values_intgr')

# subjects=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12',
#         's13','s14','s15','s16','s17','s18','s20','s21','s22','s23',
#         's24','s25','s26','s28','s29','s30','s31','s32','s33',
#         's34','s35','s36','s37'] #'s19'

subjects=['s27']

for idx_s,s in enumerate(subjects):
    print(s)
    hyp=np.loadtxt('/home/karim/projet_c1_c2_dreamer/hypnogrammes/hyp_per_{s}.txt'.format(s=s))
    hyp_norm=hyp_epoching(hyp=hyp,epoch_length=30)
    dfa_values=sio.loadmat(os.path.join(data_path,'mfdfa_{s}_diff').format(s=s))
    H=dfa_values['Hurst']
    hq_all=dfa_values['hq']
    hq_max=np.max(hq_all,axis=2)
    hq_min=np.min(hq_all,axis=2)
    M=hq_max-hq_min

    H_stages=assign_sleep_stage(H,hyp_norm)
    M_stages=assign_sleep_stage(M,hyp_norm)

    M_file = open(os.path.join(data_path,'MFDFA_stages_{}.pkl'.format(s)), "wb")
    pickle.dump(M_stages, M_file)
    M_file.close()

    H_file = open(os.path.join(data_path,'Husrt_stages_{}.pkl'.format(s)), "wb")
    pickle.dump(M_stages, H_file)
    H_file.close()
    #sio.savemat(os.path.join(data_path,'MFDFA_stages_{}.mat'.format(s)),{'H':H_stages,'M':M_stages})
