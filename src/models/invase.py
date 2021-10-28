import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.utils.validation import check_array, check_is_fitted
from torch.utils.data import DataLoader

from src.data_loading import OrganOfferDataset
from src.models.base_estimator import NeuralEstimator
from src.models.loss import Actor_Loss, Baseline_Loss, Critic_Loss
from src.models.neural_networks import INVASE


class INVASEEstimator(NeuralEstimator):
    def __init__(
        self,
        input_space,
        criteria_space,
        data_description,
        h_dim=20,
        lambda_=0.1,
        degree=1,
        num_layers=3,
        random_state=None,
        **kwargs
    ):
        self.name = "INVASE"
        self.input_space = input_space
        self.criteria_space = criteria_space
        self.data_description = data_description
        self.h_dim = h_dim
        self.lambda_ = lambda_
        self.random_state = random_state
        self.degree = degree
        self.num_layers = num_layers

    def get_params(self, deep=True):
        parameters = {
            "data_description": self.data_description,
            "input_space": self.input_space,
            "criteria_space": self.criteria_space,
            "h_dim": self.h_dim,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
            "degree": self.degree,
            "num_layers": self.num_layers,
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

    def generate_model(self):
        return INVASE(self.x_dim, self.h_dim, layer_n=self.num_layers, threshold=0.5)

    def loss(self, input, output):
        L_critic = Critic_Loss(input, output)
        L_baseline = Baseline_Loss(input, output)
        L_actor = Actor_Loss(input, output, lambda_=self.lambda_)
        return L_critic + L_baseline + L_actor

    def eval_loss(self, input, output):
        L_critic = Critic_Loss(input, output)
        return L_critic

    def predict_proba(self, X, batch_size=100):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        y = np.zeros((len(X),))

        dataset = self.create_dataset(X, y, fake_y=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        y_pred_list = []

        self._nn.eval()
        with torch.no_grad():
            for data in dataloader:
                out = self._nn(data)
                y_pred = out["prob"].detach().numpy()
                y_pred_list.append(y_pred)
        y_1 = np.concatenate(y_pred_list)
        y_0 = 1 - y_1
        return np.concatenate([y_0, y_1], axis=1)

    def score(self, X, y, threshold=0.0, sample_weight=None, batch_size=100):
        y_pred = self.predict_proba(X, threshold, batch_size)
        return average_precision_score(y, y_pred[:, 1], sample_weight=sample_weight)

    def expose_invase_output(self, X, batch_size=100, num_workers=0):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        y = np.zeros((len(X),))

        dataset = self.create_dataset(X, y, fake_y=True)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        S_list = []
        y_pred_list = []
        self._nn.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                out = self._nn(data)
                S = out["selection"].detach().numpy()
                y_pred = out["prob"].detach().numpy()
                y_pred_list.append(y_pred)
                S_list.append(S)

        S = np.concatenate(S_list)
        y_pred = np.concatenate(y_pred_list)
        return {"s": S, "action": y_pred}
