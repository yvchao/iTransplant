import torch
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score

from pkg_resources import resource_filename


def summary(metrics):
    info = [f'{metric}:{value:.3f}' for metric, value in metrics.items()]
    return ','.join(info)


def clone_state_dict(model):
    state_dict = {}
    for key in model.state_dict():
        state_dict[key] = model.state_dict()[key].clone()
    return state_dict


class Estimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super().__init__()
        self.name = "Base"

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        prob = self.predict_proba(X)
        mask = prob[:, 0] < prob[:, 1]
        y_pred = (~mask) * self.classes_[0] + mask * self.classes_[1]
        return y_pred

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict_proba(X)
        return average_precision_score(y,
                                       y_pred[:, 1],
                                       sample_weight=sample_weight)


class NeuralEstimator(Estimator):
    def __init__(self):
        super().__init__()

        self.name = "NeuralBase"

    def _train(self, optimizer, dataloader, retain_graph):
        self._nn.train()
        total_loss = 0.0
        data_size = 0
        for data in dataloader:
            optimizer.zero_grad()
            nn_out = self._nn(data)
            loss = self.loss(data, nn_out)
            loss.backward(retain_graph=retain_graph)
            optimizer.step()
            total_loss += loss.item()
            data_size += len(data['x'])
        #return total_loss/data_size
        return total_loss / len(dataloader)

    def _validation(self, dataloader):
        self._nn.eval()
        total_loss = 0.0
        data_size = 0
        with torch.no_grad():
            for data in dataloader:
                nn_out = self._nn(data)
                loss = self.eval_loss(data, nn_out)
                total_loss += loss.item()
                data_size += len(data['x'])
        #return total_loss/data_size
        return total_loss / len(dataloader)

    def fit(self,
            X,
            y,
            max_iter=10,
            batch_size=256,
            learning_rate=1e-3,
            retain_graph=False,
            tolerance=3,
            validation_split=None,
            verbose=False):
        X, y = check_X_y(X, y, accept_sparse=True)

        random_state = check_random_state(self.random_state)
        torch.manual_seed(random_state.randint(int(1e8)))

        # create data loader
        dataset = self.create_dataset(X, y)
        self.x_dim = dataset.x_dim
        self.c_dim = dataset.c_dim

        self._nn = self.generate_model()
        self.is_fitted_ = False

        self.classes_ = dataset.y_labels

        if validation_split is not None:
            sss = StratifiedShuffleSplit(n_splits=1,
                                         test_size=validation_split,
                                         random_state=random_state)
            train_set, valid_set = None, None
            for train_index, valid_index in sss.split(dataset.x, dataset.y):
                train_set = Subset(dataset, train_index)
                valid_set = Subset(dataset, valid_index)

            train_dataloader = DataLoader(train_set, batch_size=batch_size)
            valid_dataloader = DataLoader(valid_set, batch_size=batch_size)
        else:
            train_dataloader = DataLoader(dataset, batch_size=batch_size)
            valid_dataloader = None

        optimizer = optim.AdamW(self._nn.parameters(), lr=learning_rate)

        tbar = tqdm.trange(max_iter,
                           position=0,
                           leave=True,
                           disable=not verbose)
        training_history = np.full(max_iter, np.NaN)

        best_validation_loss = 1e30
        no_improvement_count = 0
        best_model = {}

        for epoch in tbar:
            train_loss = self._train(optimizer, train_dataloader, retain_graph)
            training_history[epoch] = train_loss
            metrics_train = {'train loss': train_loss}
            if valid_dataloader is not None:
                validation_loss = self._validation(valid_dataloader)
                metrics_eval = {'valid loss': validation_loss}
                if best_validation_loss > validation_loss:
                    best_validation_loss = validation_loss
                    no_improvement_count = 0
                    best_model = clone_state_dict(self._nn)
                else:
                    no_improvement_count += 1
            else:
                metrics_eval = {}

            metrics = {**metrics_train, **metrics_eval}
            tbar.set_description(summary(metrics))
            if no_improvement_count >= tolerance:
                if verbose:
                    print(
                        f'loss does not improve in {tolerance} epoches, stop training at epoch {epoch}, current loss: {validation_loss}'
                    )

                if best_model != {}:
                    self._nn.load_state_dict(best_model)
                    if verbose:
                        print('reloaded the best known model parameters')
                self.save_model(name="best")

                break

        tbar.close()

        self.is_fitted_ = True
        self.training_history = training_history

        return self

    def save_model(self, name=None):
        if name is None:
            x = datetime.datetime.now()
            date = x.strftime("%x").replace("/", "")
            clock = x.strftime("%X")
            path_tail = f"saved_models/{self.name}_{date}_{clock}.pth"
        else:
            path_tail = f"saved_models/{self.name}_{name}.pth"

        saved_models_path = resource_filename("src", "saved_models")
        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)

        path = resource_filename("src", path_tail)
        checkpoint = {
            'model_state_dict': self._nn.state_dict(),
            'x_dim': self.x_dim,
            'c_dim': self.c_dim
        }
        torch.save(checkpoint, path)

    def load_model(self, name=None):
        if name is None:
            path_tail = f"saved_models/{self.name}_best.pth"
        else:
            path_tail = f"saved_models/{self.name}_{name}.pth"
        path = resource_filename("src", path_tail)

        if os.path.exists(path):
            self.is_fitted_ = False
            checkpoint = torch.load(path)
            self.x_dim = checkpoint['x_dim']
            self.c_dim = checkpoint['c_dim']
            self._nn = self.generate_model()
            self._nn.load_state_dict(checkpoint['model_state_dict'])
            self.is_fitted_ = True
        else:
            raise Exception(f'no model found in {path}')
