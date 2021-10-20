from src.models.base_estimator import Estimator
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from src.data_loading import OrganOfferDataset

class LowessEstimator(Estimator):
    def __init__(self,input_space, criteria_space, data_description,tau=1.0,degree=1,random_state=None, **kwargs):
        self.name = "LOWESS"
        self.input_space=input_space
        self.criteria_space=criteria_space
        self.data_description=data_description
        self.tau=tau
        self.random_state=random_state
        self.degree=degree

    def get_params(self, deep=True):
        parameters={
            'input_space': self.input_space,
            'criteria_space': self.criteria_space,
            'data_description': self.data_description,
            'tau':self.tau,
            'random_state':self.random_state,
            'degree': self.degree,
            }
            
        return parameters

    def create_dataset(self,X,y, fake_y=False):
        return OrganOfferDataset(X,y,self.input_space,self.criteria_space,self.data_description,degree=self.degree,fake_y=fake_y)

    def fit(self, X, y,**kwargs):
        X, y = check_X_y(X, y, accept_sparse=True)

        dataset=self.create_dataset(X,y)
        self.is_fitted_ = False

        self.classes_ = dataset.y_labels

        self._x_train=dataset.x
        self._c_train=dataset.c
        self._y_train=dataset.y

        self.is_fitted_ = True
        return self

    def loess(self,x0,c0):
        x=self._x_train
        c=self._c_train
        y=self._y_train[:,np.newaxis]
        cc=c[:,:,np.newaxis]@c[:,np.newaxis,:]
        w=np.exp(-(np.linalg.norm(x0-x,ord=2,axis=1,keepdims=True)/self.tau)**2/2)
        b=np.bmat([[np.sum(w * y,axis=0)], [np.sum(w * y * c,axis=0)]]).T
        A=np.bmat([[np.sum(w,axis=0)[:,np.newaxis],np.sum(w * y * c,axis=0)[np.newaxis,:]],[np.sum(w * c,axis=0)[:,np.newaxis],np.sum(w[:,:,np.newaxis] * cc,axis=0)]])
        beta = np.linalg.lstsq(A,b,rcond=None)[0]
        beta=beta.A[:,0]
        y0_pred=beta[0]+beta[1:]@c0[0]
        return y0_pred,beta


    def get_feature_importance(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        y=np.zeros((len(X),))

        dataset=self.create_dataset(X,y, fake_y=True)
        X=dataset.x
        C=dataset.c
        if len(X.shape)==1:
            X.reshape((1,-1))
        if len(C.shape)==1:
            C.reshape((1,-1))

        W_list=[]
        for i in range(len(X)):
            _,w=self.loess(X[[i]],C[[i]])
            W_list.append(w.reshape(1,-1))
        W=np.concatenate(W_list)
        return W[:,1:],W[:,0]

    def get_counterfactual_predict(self, X, W,W0):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        y=np.zeros((len(X),))

        dataset=self.create_dataset(X,y, fake_y=True)
        criteria=dataset.c
        return np.clip(W0+np.sum(W*criteria,axis=-1),0,1)

    def predict_proba(self,X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        y=np.zeros((len(X),))

        dataset=self.create_dataset(X,y, fake_y=True)
        X=dataset.x
        C=dataset.c
        if len(X.shape)==1:
            X.reshape((1,-1))
        if len(C.shape)==1:
            C.reshape((1,-1))
        y_pred=np.zeros((len(X),2))

        for i in range(len(X)):
            prob,_=self.loess(X[[i]],C[[i]])
            prob=np.clip(prob,0,1)
            y_pred[i,0]=1-prob
            y_pred[i,1]=prob
      
        return y_pred

   
    