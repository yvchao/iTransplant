import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.data_loading import OrganOfferDataset
from src.models.base_estimator import Estimator


class DecisionTreeEstimator(Estimator):
    def __init__(
        self,
        input_space,
        criteria_space,
        data_description,
        max_depth=10,
        n_clusters=10,
        degree=1,
        random_state=None,
        **kwargs
    ):
        self.name = "Decision Tree"
        self.input_space = input_space
        self.criteria_space = criteria_space
        self.data_description = data_description
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_depth = max_depth
        self.degree = degree

    def get_params(self, deep=True):
        parameters = {
            "input_space": self.input_space,
            "criteria_space": self.criteria_space,
            "data_description": self.data_description,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "max_depth": self.max_depth,
            "degree": self.degree,
        }

        return parameters

    def create_decision_tree(self, random_state, max_depth):
        return DecisionTreeClassifier(
            random_state=random_state, class_weight="balanced", max_depth=max_depth
        )

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
        patient_features = dataset.x
        criteria = dataset.c

        self.regressors = [
            self.create_decision_tree(random_state, self.max_depth)
            for i in range(self.n_clusters)
        ]

        self.cls = KMeans(n_clusters=self.n_clusters, random_state=random_state)
        c_labels = self.cls.fit_predict(patient_features)
        for i in range(self.n_clusters):
            mask = c_labels == i
            Xi = criteria[mask]
            yi = y[mask]
            if len(np.unique(yi)) < 2:
                self.regressors[i] = None
            else:
                self.regressors[i].fit(Xi, yi)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        y = np.zeros((len(X),))

        dataset = self.create_dataset(X, y, fake_y=True)
        patient_features = dataset.x
        criteria = dataset.c

        y_pred = np.zeros((len(X), 2))
        c_labels = self.cls.predict(patient_features)
        for i in range(self.n_clusters):
            mask = c_labels == i
            Xi = criteria[mask]
            if len(Xi) < 1:
                continue
            if self.regressors[i] is None:
                y_pred[mask] = 0.5
            else:
                y_pred[mask] = self.regressors[i].predict_proba(Xi)

        return y_pred
