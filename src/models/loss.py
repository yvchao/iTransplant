import numpy as np
import torch
import torch.nn.functional as F


def MoE_Loss(input, output):
    moe_loss = torch.squeeze(output["moe_loss"])
    return moe_loss


def BCE_Loss(input, output):
    a = input["y"].view(-1, 1)
    a_pred = output["prob"]
    w = input["sample_weight"].view(-1, 1)
    return F.binary_cross_entropy(a_pred, a, weight=w)


def AE_Loss(input, net_out):
    X = input["x"]
    X_rec = net_out["x_rec"]
    return F.mse_loss(X_rec, X)


def Baseline_Loss(data_in, net_out):
    a = data_in["y"].view(-1, 1)
    a_pred = net_out["p_baseline"]
    w = data_in["sample_weight"].view(-1, 1)
    return F.binary_cross_entropy(a_pred, a, weight=w)


def Critic_Loss(data_in, net_out):
    a = data_in["y"].view(-1, 1)
    a_pred = net_out["p_critic"]
    w = data_in["sample_weight"].view(-1, 1)
    return F.binary_cross_entropy(a_pred, a, weight=w)


def Actor_Loss(data_in, net_out, lambda_=0.1, eps=1e-8):
    a = data_in["y"].view(-1, 1)
    a_critic = net_out["p_critic"]
    a_baseline = net_out["p_baseline"]
    w = data_in["sample_weight"].view(-1, 1)
    critic_loss = F.binary_cross_entropy(
        a_critic, a, weight=w, reduction="sum"
    ).detach()
    baseline_loss = F.binary_cross_entropy(
        a_baseline, a, weight=w, reduction="sum"
    ).detach()
    Reward = critic_loss - baseline_loss
    p_selection = net_out["p_selection"]
    selection = net_out["selection"]
    actor_loss = (
        Reward
        * torch.sum(
            selection * torch.log(p_selection + eps)
            + (1 - selection) * torch.log(1 - p_selection + eps),
            dim=1,
        )
        + lambda_ * torch.mean(p_selection, dim=1)
    )
    return torch.mean(actor_loss)


def pairwise_distances(x):
    """
    Input: x is a Nxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    dist = dist - torch.diag(dist.diag())
    return torch.clamp(dist, 0.0, np.inf)


def Lipschitz(input, output, eps=1e-6):
    Z_dist = pairwise_distances(output["z"].detach())
    W_dist = pairwise_distances(output["w"])
    return 0.5 * torch.mean(W_dist / (Z_dist + eps))
