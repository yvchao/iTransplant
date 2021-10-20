from src.models.base_estimator import Estimator
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from src.data_loading import OrganOfferDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state


class RandomForestEstimator(Estimator):
    def __init__(self,input_space, criteria_space, data_description,n_estimators=10,max_depth=10,degree=1,random_state=None, **kwargs):
        self.name = "Random Forest"
        self.input_space=input_space
        self.criteria_space=criteria_space
        self.data_description=data_description
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.random_state=random_state
        self.degree=degree

    def get_params(self, deep=True):
        parameters={
            'input_space': self.input_space,
            'criteria_space': self.criteria_space,
            'data_description': self.data_description,
            'n_estimators':self.n_estimators,
            'max_depth': self.max_depth,
            'random_state':self.random_state,
            'degree': self.degree,
            }
            
        return parameters

    def create_dataset(self,X,y, fake_y=False):
        return OrganOfferDataset(X,y,self.input_space,self.criteria_space,self.data_description,degree=self.degree,fake_y=fake_y)

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, accept_sparse=True)
        random_state = check_random_state(self.random_state)
        self.is_fitted_ = False

        dataset=self.create_dataset(X,y)
        self.classes_ = dataset.y_labels
        features=dataset.x

        self.clf=RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,random_state=random_state,class_weight='balanced')
        self.clf.fit(features, y)

        self.is_fitted_ = True
        return self

    def predict_proba(self,X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        y=np.zeros((len(X),))

        dataset=self.create_dataset(X,y, fake_y=True)
        features=dataset.x

        y_pred=self.clf.predict_proba(features)
      
        return y_pred
