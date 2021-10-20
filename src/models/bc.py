from src.models.base_estimator import NeuralEstimator
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from src.data_loading import OrganOfferDataset
from sklearn.metrics import average_precision_score

from src.models.neural_networks import BlackBox
from src.models.loss import BCE_Loss


class BCEstimator(NeuralEstimator):
    def __init__(self,
                 input_space,
                 criteria_space,
                 data_description,
                 h_dim=20,
                 degree=1,
                 num_layers=3,
                 random_state=None,
                 **kwargs):
        self.name = "BC"
        self.input_space = input_space
        self.criteria_space = criteria_space
        self.data_description = data_description
        self.h_dim = h_dim
        self.random_state = random_state
        self.degree = degree
        self.num_layers = num_layers

    def get_params(self, deep=True):
        parameters = {
            'data_description': self.data_description,
            'input_space': self.input_space,
            'criteria_space': self.criteria_space,
            'h_dim': self.h_dim,
            'random_state': self.random_state,
            'degree': self.degree,
            'num_layers': self.num_layers
        }

        return parameters

    def create_dataset(self, X, y, fake_y=False):
        return OrganOfferDataset(X,
                                 y,
                                 self.input_space,
                                 self.criteria_space,
                                 self.data_description,
                                 degree=self.degree,
                                 fake_y=fake_y)

    def generate_model(self):
        return BlackBox(self.x_dim, self.h_dim, layer_n=self.num_layers)

    def loss(self, input, output):
        L_policy = BCE_Loss(input, output)
        return L_policy

    def eval_loss(self, input, output):
        return self.loss(input, output)

    def predict_proba(self, X, batch_size=100):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        y = np.zeros((len(X), ))

        dataset = self.create_dataset(X, y, fake_y=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        y_pred_list = []

        self._nn.eval()
        with torch.no_grad():
            for data in dataloader:
                out = self._nn(data)
                y_pred = out['prob'].detach().numpy()
                y_pred_list.append(y_pred)
        y_1 = np.concatenate(y_pred_list)
        y_0 = 1 - y_1
        return np.concatenate([y_0, y_1], axis=1)

    def score(self, X, y, threshold=0.0, sample_weight=None, batch_size=100):
        y_pred = self.predict_proba(X, threshold, batch_size)
        return average_precision_score(y,
                                       y_pred[:, 1],
                                       sample_weight=sample_weight)
