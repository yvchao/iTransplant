import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.utils.validation import check_array, check_is_fitted
from torch.utils.data import DataLoader

from src.data_loading import OrganOfferDataset
from src.models.base_estimator import NeuralEstimator
from src.models.loss import AE_Loss, BCE_Loss, Lipschitz, MoE_Loss
from src.models.neural_networks import iTransplant


class iTransplantEstimator(NeuralEstimator):
    def __init__(
        self,
        input_space,
        criteria_space,
        data_description,
        h_dim=20,
        num_experts=10,
        k=4,
        degree=1,
        lambda_=0.01,
        num_layers=3,
        random_state=None,
        **kwargs
    ):
        self.name = "iTransplant"
        self.input_space = input_space
        self.criteria_space = criteria_space
        self.data_description = data_description
        self.h_dim = h_dim
        self.lambda_ = lambda_
        self.num_experts = num_experts
        self.k = k
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
            "num_experts": self.num_experts,
            "k": self.k,
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
        return iTransplant(
            self.x_dim,
            self.c_dim,
            self.h_dim,
            layer_n=self.num_layers,
            num_experts=self.num_experts,
            k=self.k,
            sigma=torch.sigmoid,
            threshold=0,
        )

    def loss(self, input, output):
        L_policy = BCE_Loss(input, output)
        L_reconstruction = AE_Loss(input, output)
        L_MoE = MoE_Loss(input, output)
        L_Consistency = Lipschitz(input, output)
        return L_policy + L_reconstruction + L_MoE + L_Consistency * self.lambda_

    def eval_loss(self, input, output):
        L_policy = BCE_Loss(input, output)
        return L_policy

    def predict_proba(self, X, threshold=0.0, batch_size=100):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        y = np.zeros((len(X),))

        dataset = self.create_dataset(X, y, fake_y=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        y_pred_list = []

        self._nn.threshold = threshold

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

    def get_feature_importance(self, X, batch_size=100):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        y = np.zeros((len(X),))

        dataset = self.create_dataset(X, y, fake_y=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        W_list = []
        self._nn.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                out = self._nn(data)
                W = out["w"].detach().numpy()
                W_list.append(W)

        W = np.concatenate(W_list)
        return W, np.zeros(len(W))

    def get_counterfactual_predict(self, X, W, W0):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        y = np.zeros((len(X),))

        dataset = self.create_dataset(X, y, fake_y=True)
        criteria = dataset.c
        return 1 / (1 + np.exp(-np.sum(W * criteria, axis=-1)))

    def expose_nn_output(self, X, batch_size=100):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        y = np.zeros((len(X),))

        dataset = self.create_dataset(X, y, fake_y=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        W_list = []
        Z_list = []
        y_pred_list = []
        gate_list = []
        self._nn.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                out = self._nn(data)
                W = out["w"].detach().numpy()
                Z = out["z"].detach().numpy()
                y_pred = out["prob"].detach().numpy()
                gates = out["gates"].detach().numpy()
                y_pred_list.append(y_pred)
                W_list.append(W)
                Z_list.append(Z)
                gate_list.append(gates)

        W = np.concatenate(W_list)
        Z = np.concatenate(Z_list)
        y_pred = np.concatenate(y_pred_list)
        gates = np.concatenate(gate_list)
        return {"w": W, "z": Z, "action": y_pred, "gates": gates}
