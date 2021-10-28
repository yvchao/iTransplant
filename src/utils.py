import os

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.data_loading import aggregate_data, create_dataset


def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    elif not os.path.isdir(dir):
        raise Exception(f"dir {dir} is not a valid value.")
    else:
        pass


def create_estimator(
    model, data_dict, column_dict, random_state=None, hyper_parameter={}
):
    transformers = []
    for k, v in column_dict.items():
        transformers.append((f"num_{k}", RobustScaler(quantile_range=(5, 95)), v[0]))
        transformers.append((f"cat_{k}", FunctionTransformer(), v[1]))
    preprocessor = ColumnTransformer(transformers=transformers)

    estimator = model(
        data_description=data_dict, random_state=random_state, **hyper_parameter
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return clf, [preprocessor, estimator]


def train_evaluate(
    model,
    data_dir,
    centers,
    seed,
    scoring,
    verbose=False,
    discretize=False,
    max_iter=100,
    validation_split=0.2,
    hyper_parameter={},
):
    aggregate_data_source = aggregate_data(data_dir, centers)
    X, y, data_dict, column_dict = create_dataset(aggregate_data_source, discretize)

    clf, _ = create_estimator(
        model,
        data_dict,
        column_dict,
        random_state=seed,
        hyper_parameter=hyper_parameter,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40108642, stratify=y
    )
    clf.fit(
        X_train,
        y_train,
        estimator__max_iter=max_iter,
        estimator__verbose=verbose,
        estimator__validation_split=validation_split,
    )

    scores = {"model": clf.named_steps["estimator"].name}
    for k, scorer in scoring.items():
        scores[k] = scorer(clf, X_test, y_test)
    return scores


def inspection(
    model,
    data_dir,
    centers,
    seed,
    discretize=False,
    max_iter=100,
    validation_split=0.2,
    hyper_parameter={},
):
    aggregate_data_source = aggregate_data(data_dir, centers)
    X, y, data_dict, column_dict = create_dataset(aggregate_data_source, discretize)

    clf, _ = create_estimator(
        model,
        data_dict,
        column_dict,
        verbose=False,
        random_state=seed,
        hyper_parameter=hyper_parameter,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40108642, stratify=y
    )
    clf.fit(
        X_train,
        y_train,
        estimator__max_iter=max_iter,
        estimator__validation_split=validation_split,
    )
    probs = clf.predict_proba(X_test)
    return probs, y_test, clf
