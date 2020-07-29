from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
#from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RF
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import permutation_test_score
import scipy.io as sio
#import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from mne import pick_types, find_layout
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV

def get_data(list1=[],list2=[],group=[],data_path=None,C=[],stade=None,moy=0,p=2,add_pe=False):
    Dr_data=[]
    nDr_data=[]
    y=[]
    for s1,s2 in zip (range(list1.shape[0]),range(list2.shape[0])):
        M1=load_C1C2(subject=list1[s1],C=C,stade=stade,moy=moy,path=data_path,p=p)
        Dr_data.append(M1)
        M2=load_C1C2(subject=list2[s2],C=C,stade=stade,moy=moy,path=data_path,p=p)
        nDr_data.append(M2)

        #y.append([1]*len(Dr_data)+[0]*len(nDr_data))

    if add_pe:
        pndr,pdr=np.array([]),np.array([])
        print('loading Permutation entropy: ')
        for s1,s2 in zip(range(sDreamer.shape[0]),range(sNnDreamer.shape[0])):
            p1=load_pe(path=pe_path,subject=sDreamer[s1],stade=st,moy=moy)
            pdr=np.concatenate((pdr,p1),axis=0) if pdr.size else p1
            p2=load_pe(path=pe_path,subject=sNnDreamer[s2],stade=st,moy=moy)
            pndr=np.concatenate((pndr,p2),axis=0) if pndr.size else p2
        xpe=np.vstack((pdr,pndr))

    X=Dr_data+nDr_data

    y_ndr=[np.asarray([0]*sub.shape[0]) for sub in  nDr_data]
    y_dr=[np.asarray([1]*sub.shape[0]) for sub in  Dr_data]
    y=y_dr+y_ndr

    groups=[np.asarray([group[ind]]*sub.shape[0]) for ind,sub in enumerate(X)]

    sz_ndr = [sub.shape[0] for sub in  nDr_data]
    sz_dr = [sub.shape[0] for sub in  Dr_data]
    sizes=np.concatenate((sz_dr,sz_ndr),axis=0)
    return X,y,groups,sizes

def load_C1C2(subject=[],C=None,stade=None,moy=1,path=None,p=2):

    file='/lin_C1C2C3_guy{s}_{st}_{cl}.mat'
    #file = '/{c}/C1C2C3_S{s}_{st}_{c}.mat'
    ck = sio.loadmat(path+file.format(s=subject,st=stade,cl=p))[C][0:19,:]
    if moy:
        mat=np.mean(ck,axis=1)
        M=np.reshape(mat,(1,19))
    else:
        M=ck.T

    return np.array(M)
def load_pe(path=None,subject=[],stade=None,moy=0):
    file='permut_s{}.mat'.format(str(subject))
    M=sio.loadmat(path+file)['permut_{}'.format(stade)][:,0:19]
    if moy:
        M=np.reshape(np.mean(M,axis=0),(1,19))
    return M

def get_classifier(clf_name=None, inner_cv=None):
    clf = None
    fit_params = {}
    if clf_name=='logreg':
        if inner_cv==None:
            clf_init = LogisticRegression(random_state=0)
            clf=Pipeline(StandardScaler(),clf_init)
        else:
            clf_init=LogisticRegression(random_state=0,solver='liblinear')

            random_grid = {'penalty':['l1','l2'] ,
                           'C': np.logspace(-4, 4, 20)}
            n_iter_search=40

            Rnd_Srch = RandomizedSearchCV(clf_init, param_distributions=random_grid,
                                           n_iter=n_iter_search, cv=inner_cv,iid=True)
            clf=make_pipeline(RobustScaler(),Rnd_Srch)#,PCA(n_components=5)

    if clf_name=='LDA':
        clf=LDA()

    if clf_name=='RBF_svm':
        svm=SVC(kernel='rbf')
        # parameters for grid search
        if inner_cv==None:
            clf=svm

        else:
            p_grid = {}
            p_grid['gamma']= [1e-3, 1e-4]
            p_grid['C']= [1, 10, 100, 1000]
            # classifier
            #clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
            n_iter_search=10
            Rnd_Srch = RandomizedSearchCV(svm, param_distributions=p_grid,
                                           n_iter=n_iter_search, cv=inner_cv)
            clf = make_pipeline(StandardScaler(),Rnd_Srch)
    elif clf_name == 'linear_svm_scaled':
        svm = SVC(kernel='linear')
        # parameters for grid search
        if inner_cv=='None':
            clf=svm
        else:
            p_grid = {}
            p_grid['C'] = np.power(10.0, np.linspace(-4, 4, 25))
            # classifier
            n_iter_search=10
            Rnd_Srch = RandomizedSearchCV(svm, param_distributions=p_grid,
                                           n_iter=n_iter_search, cv=inner_cv)
            clf = make_pipeline(StandardScaler(),Rnd_Srch)
    elif clf_name == 'RF':
        if inner_cv==None:
            clf=RF()

        else:
            clf_init=RF()
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 50)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth}
            n_iter_search=10

            Rnd_Srch = RandomizedSearchCV(clf_init, param_distributions=random_grid,
                                           n_iter=n_iter_search, cv=inner_cv,iid=True)
            #clf=Rnd_Srch
            clf = make_pipeline(StandardScaler(),Rnd_Srch)

    return clf
#def run_features_selection(method='None',inner_cv=None,outer_cv=None):

def get_column(List, i):
    return [np.asarray(row[:,i]) for row in List]

def run_classification(clf_name=None,
                        X=[],
                        y=[],
                        groups=None,
                        inner_cv=None,
                        outer_cv=None,
                        mode='mf',
                        stat=None,
                        get_estimator=False,
                        nb_permutations=100,
                        n_jobs=-1):
    """ Run classification procedure in multi or single feature

    Parameters
    ----------

    *clf: classifier name
    X=Feature matrice, should be n_samples x n_features
    y=Labels vector
    groups:
    stat : if true run permutations test
    nb_permutations=number of permutations to run if stat is True
    mode: string/by default = 'mf'; could be
        - 'mf' for multifeature classification without feature selection
        - 'mf_fs' for multifeature classification with feature selection
        - 'sf' for single feature classification
        - 'mf_sg_elect' for multifeature single electrode
    """
    if isinstance(X, list):
        n_samples,n_features=X[0].shape

        n_electrodes=len(X)
    else:
        n_samples,n_features=X.shape

    if mode=='sf':

        test_scores=[]
        permutation_scores=[]
        pvalues=[]

        for feat in range(n_features):
            clf= get_classifier(clf_name,inner_cv)
            if stat:

                print('\n for feature num: \n' , feat)
                test_score, permutation_score, pvalue = permutation_test_score(
                                                        clf, X[:,feat].reshape(-1,1), y,
                                                        scoring="accuracy",
                                                        cv=outer_cv,
                                                        groups=groups,
                                                        n_permutations=nb_permutations,
                                                        n_jobs=n_jobs)

                print("Test accuracy = %0.4f "%(test_score))
                test_scores.append(test_score)
                permutation_scores.append(permutation_score)
                pvalues.append(pvalue)



            else:
                print('Running classification without permutations')
                print('\n for feature num: \n' , feat+1)
                output = cross_validate(clf,
                                        X = X[:,feat].reshape(-1,1),
                                        y = y,
                                        cv = outer_cv,
                                        return_train_score = True,
                                        scoring='accuracy',
                                        n_jobs = n_jobs)
                                        #fit_params=fit_params,


                test_scores.append(output['test_score'].mean())
    if mode=='mf_sg_elect':
        test_scores=[]
        permutation_scores=[]
        pvalues=[]

        for elect in range(n_electrodes):
            clf= get_classifier(clf_name,inner_cv)
            if stat:

                print('\n for electrode:' , elect+1)
                print(X[elect].shape)
                test_score, permutation_score, pvalue = permutation_test_score(
                                                        clf, X[elect], y,
                                                        scoring="accuracy",
                                                        cv=outer_cv,
                                                        n_permutations=nb_permutations,
                                                        n_jobs=n_jobs)

                print("Test accuracy = %0.4f "%(test_score))
                test_scores.append(test_score)
                permutation_scores.append(permutation_score)
                pvalues.append(pvalue)

    elif mode=='mf': #(multi feature without feature selection)
        test_scores=[]
        permutation_scores=[]
        pvalues=[]
        clf= get_classifier(clf_name,inner_cv)

        if stat:
            print('Running multi feature classification with permutations')
            test_score, permutation_score, pvalue = permutation_test_score(
                                                    clf, X, y,
                                                    scoring="accuracy",
                                                    cv=outer_cv,
                                                    n_permutations=nb_permutations,
                                                    n_jobs=n_jobs)

            print("Test accuracy = %0.4f "%(test_score))
            test_scores.append(test_score)
            permutation_scores.append(permutation_score)
            pvalues.append(pvalue)

        else:
            print('Running Multifeature classification without permutations')

            output = cross_validate(clf,
                                    X = X,
                                    y = y,
                                    cv = outer_cv,
                                    return_train_score = True,
                                    scoring='accuracy',
                                    return_estimator=True,
                                    n_jobs = n_jobs)

            if get_estimator:
                reslt=dict()
                models=output['estimator']
                coef=[]

                for model in models:
                    if 'randomizedsearchcv' in model.named_steps:
                        coef.append(model.named_steps['randomizedsearchcv'].best_estimator_.coef_)
                    else:
                        coef.append(model.named_steps['reg'].coef_)
                results.update({'coeff':coef})

            #print("Test accuracy = %0.4f +- %0.4f"%(output['test_accuracy'].mean(), output['test_accuracy'].std()))
            test_scores.append(output['test_score'].mean())


    # elif mode=='feature_selection':
    #     if method=='RFECV':


    return test_scores,permutation_scores,pvalues

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
#
class StratifiedShuffleGroupSplit(BaseEstimator):
    def __init__(self, n_groups, n_iter=None):
        self.n_groups = n_groups
        self.n_iter = n_iter
        self.counter = 0
        self.labels_list = []
        self.n_each = None
        self.n_labs = None
        self.labels_list = None
        self.lpgos = None
        self.indexes = None

    def _init_atributes(self, y, groups):
        if len(y) != len(groups):
            raise Exception("Error: y and groups need to have the same length")
        if y is None:
            raise Exception("Error: y cannot be None")
        if groups is None:
            raise Exception("Error: this function requires a groups parameter")
        if self.labels_list is None:
            self.labels_list = list(set(y))
        if self.n_labs is None:
            self.n_labs = len(self.labels_list)
        assert (
            self.n_groups % self.n_labs == 0
        ), "Error: The number of groups to leave out must be a multiple of the number of classes"
        if self.n_each is None:
            self.n_each = int(self.n_groups / self.n_labs)
        if self.lpgos is None:
            lpgos, indexes = [], []
            for label in self.labels_list:
                index = np.where(y == label)[0]
                indexes.append(index)
                lpgos.append(LeavePGroupsOut(self.n_each))
            self.lpgos = lpgos
            self.indexes = np.array(indexes)

    def split(self, X, y, groups):
        self._init_atributes(y, groups)
        y = np.asarray(y)
        groups = np.asarray(groups)
        iterators = []
        for lpgo, index in zip(self.lpgos, self.indexes):
            iterators.append(lpgo.split(index, y[index], groups[index]))
        for ite in product(*iterators):
            if self.counter == self.n_iter:
                break
            self.counter += 1
            train_idx = np.concatenate(
                [index[it[0]] for it, index in zip(ite, self.indexes)]
            )
            test_idx = np.concatenate(
                [index[it[1]] for it, index in zip(ite, self.indexes)]
            )
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups):
        self._init_atributes(y, groups)
        if self.n_iter is not None:
            return self.n_iter
        groups = np.asarray(groups)
        n = 1
        for index, lpgo in zip(self.indexes, self.lpgos):
            n *= lpgo.get_n_splits(None, None, groups[index])
        return n
