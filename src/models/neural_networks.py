import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.mlp import MLP
from src.models.moe import MoE


class iTransplant(nn.Module):
    def __init__(
        self,
        x_dim,
        c_dim,
        h_dim=20,
        layer_n=3,
        num_experts=5,
        k=4,
        sigma=torch.sigmoid,
        threshold=0,
    ):
        super().__init__()
        self.c_dim = c_dim

        self.encoder = MLP(x_dim, h_dim, h_dim, layer_n)
        self.decoder = MLP(h_dim, x_dim, h_dim, layer_n)
        self.policy_mapping = MoE(h_dim, c_dim, num_experts, h_dim, k=k)
        self.sigma = sigma
        # self.w0=nn.Parameter(torch.randn(1))
        self.threshold = threshold

    def forward(self, input):
        X = input["x"]
        criteria = input["c"]
        Z = self.encoder(X)
        X_hat = self.decoder(Z)
        w, gates, MoE_loss = self.policy_mapping(Z, train=self.training)

        if not self.training:
            mask = torch.abs(w) >= self.threshold * torch.norm(w, dim=1, keepdim=True)
            w = w * mask

        score = torch.sum(criteria * w, dim=-1, keepdim=True)  # +self.w0
        prob = self.sigma(score)

        out = {}

        out["w"] = w
        out["prob"] = prob
        out["z"] = Z
        out["x_rec"] = X_hat
        out["moe_loss"] = MoE_loss
        out["gates"] = gates
        return out


class BlackBox(nn.Module):
    def __init__(self, x_dim, h_dim=20, layer_n=3):
        super().__init__()
        self.nn = nn.Sequential(MLP(x_dim, 1, h_dim, layer_n), nn.Sigmoid())

    def forward(self, input):
        X = input["x"]
        prob = self.nn(X)
        out = {"prob": prob}
        return out


class AutoEncoder(nn.Module):
    def __init__(self, x_dim, z_dim=20, h_dim=20, layer_n=3):
        super().__init__()

        self.encoder = MLP(x_dim, z_dim, h_dim, layer_n)
        self.decoder = MLP(z_dim, x_dim, h_dim, layer_n)

    def forward(self, input):
        X = input["x"]
        Z = self.encoder(X)
        X_hat = self.decoder(Z)

        out = {}

        out["z"] = Z
        out["x_rec"] = X_hat
        return out


def bernoulli_sampling(prob):
    n, d = prob.shape
    samples = np.random.binomial(1, prob, (n, d))
    return samples.astype("float32")


class INVASE(nn.Module):
    def __init__(self, x_dim, h_dim=20, layer_n=3, threshold=0.5):
        super().__init__()

        self.baseline = nn.Sequential(MLP(x_dim, 1, h_dim, layer_n), nn.Sigmoid())
        self.critic = nn.Sequential(MLP(x_dim, 1, h_dim, layer_n), nn.Sigmoid())
        self.selector = nn.Sequential(MLP(x_dim, x_dim, h_dim, layer_n), nn.Sigmoid())

        self.threshold = threshold

    def forward(self, input):
        X = input["x"]
        p_baseline = self.baseline(X)
        p_selection = self.selector(X)
        if self.training == True:
            selection = torch.from_numpy(bernoulli_sampling(p_selection.detach()))
        else:
            selection = (1.0 * (p_selection > self.threshold)).detach()
        p_critic = self.critic(X * selection)

        out = {}
        out["prob"] = p_critic
        out["p_baseline"] = p_baseline
        out["p_critic"] = p_critic
        out["p_selection"] = p_selection
        out["selection"] = selection
        return out
