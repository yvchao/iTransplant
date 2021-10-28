import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import Dataset


class OrganOfferDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        input_space,
        criteria_space,
        data_dict,
        degree=1,
        transform=None,
        fake_y=False,
    ):
        patient_features = X[:, data_dict["X"]["patient"]].astype("float32")
        organ_features = X[:, data_dict["X"]["organ"]].astype("float32")
        criteria = X[:, data_dict["X"]["criteria"]].astype("float32")

        if input_space == "XxC":
            self.x = np.concatenate([patient_features, criteria], axis=1)
        elif input_space == "XxO":
            self.x = np.concatenate([patient_features, organ_features], axis=1)
        elif input_space == "C":
            self.x = criteria
        elif input_space == "X":
            self.x = patient_features
        elif input_space == "O":
            self.x = organ_features
        else:
            raise Exception(f"input space {input_space} cannot be understood.")

        if criteria_space == "C":
            self.c = criteria
        elif criteria_space == "O":
            self.c = organ_features
        else:
            raise Exception(f"criteria space {criteria_space} cannot be understood.")

        poly = PolynomialFeatures(degree, include_bias=False)
        self.c = poly.fit_transform(self.c)

        self.y_labels = data_dict["y_labels"]

        self.y = 1.0 * (y == self.y_labels[1]).astype("float32")

        if fake_y:
            self.sample_weight = np.ones_like(self.y).astype("float32")
        else:
            assert len(unique_labels(y)) == 2
            # N = len(self.y)
            # mask_0 = self.y==0
            # N_0 = np.sum(mask_0)
            # assert N_0>0 and N_0<N
            # self.sample_weight=(mask_0*0.5*N/N_0+(1-mask_0)*0.5*N/(N-N_0)).astype('float32')
            mask = self.y == 0
            p_neg = np.sum(mask) / len(mask)
            assert p_neg > 0 and p_neg < 1
            self.sample_weight = (mask * (1 - p_neg) + (1 - mask) * p_neg).astype(
                "float32"
            )

        self.length = len(self.y)
        self.x_dim = self.x.shape[1]
        self.c_dim = self.c.shape[1]

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "x": self.x[idx],
            "c": self.c[idx],
            "y": self.y[idx],
            "sample_weight": self.sample_weight[idx],
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
