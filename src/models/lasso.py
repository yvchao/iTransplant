import numpy as np
from sklearn.linear_model import Lasso
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.data_loading import OrganOfferDataset
from src.models.base_estimator import Estimator


class LASSOEstimator(Estimator):
    def __init__(
        self,
        input_space,
        criteria_space,
        data_description,
        alpha=1.0,
        degree=1,
        random_state=None,
        **kwargs
    ):
        self.name = "LASSO"
        self.input_space = input_space
        self.criteria_space = criteria_space
        self.data_description = data_description
        self.alpha = alpha
        self.random_state = random_state
        self.degree = degree

    def get_params(self, deep=True):
        parameters = {
            "input_space": self.input_space,
            "criteria_space": self.criteria_space,
            "data_description": self.data_description,
            "alpha": self.alpha,
            "random_state": self.random_state,
            "degree": self.degree,
        }

        return parameters

    def create_dataset(self, X, y, fake_y=False):
        return OrganOfferDataset(
            X,
            y,
            self.input_space,
            self.criteria_space,
            self.data_description,
            degree=self.degree,
            fake_y=fake_y,
        )

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, accept_sparse=True)

        random_state = check_random_state(self.random_state)
        self.is_fitted_ = False

        dataset = self.create_dataset(X, y)
        self.classes_ = dataset.y_labels
        X = dataset.x
        sample_weight = dataset.sample_weight

        self.clf = Lasso(alpha=self.alpha, random_state=random_state)
        self.clf.fit(X, y, sample_weight=sample_weight)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        y = np.zeros((len(X),))
        dataset = self.create_dataset(X, y, fake_y=True)
        X = dataset.x
        y_pred = np.zeros((len(X), 2))
        y_pred[:, 1] = np.clip(self.clf.predict(X), 0, 1)
        y_pred[:, 0] = 1 - y_pred[:, 1]

        return y_pred
