import os

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data_loading import aggregate_data, create_dataset
from src.data_loading.data_utils import (
    aggregate_data,
    create_criteria,
    create_dataset,
    meld,
    meld_na,
)
from src.models import BCEstimator, INVASEEstimator, iTransplantEstimator
from src.utils import create_dir_if_not_exist, create_estimator

torch.use_deterministic_algorithms(True)

numerical_features = [
    "AGE",
    "HGT_CM_CALC",
    "WGT_KG_CALC",
    "BMI_CALC",
    "CREAT_TX",
    "INR_TX",
    "TBILI_TX",
    "MELD_PELD_LAB_SCORE",
    "FINAL_SERUM_SODIUM",
    "ALBUMIN_TX",
    "NUM_PREV_TX",
    "AGE_DON",
    "HGT_CM_DON_CALC",
    "WGT_KG_DON_CALC",
    "TBILI_DON",
    "CREAT_DON",
    "SGOT_DON",
    "SGPT_DON",
]


def perturb_samples(
    X_train, X_test, perturbed_features, perturb_signs, data_dict, scale=0.1
):
    X_tr = X_train[:, data_dict["X"]["patient"]]
    O_tr = X_train[:, data_dict["X"]["organ"]]
    X_te = X_test[:, data_dict["X"]["patient"]]
    X_purterb = X_test[:, data_dict["X"]["patient"]].copy()
    O_purterb = X_test[:, data_dict["X"]["organ"]].copy()

    patient_features = data_dict["patient_features"]
    organ_features = data_dict["organ_features"]

    for f, perturb_sign in zip(perturbed_features, perturb_signs):
        is_num = f in numerical_features

        if f in patient_features:
            ([idx],) = np.where(patient_features == f)
            if is_num:
                std = np.std(X_tr[:, idx])
                X_purterb[:, idx] = X_purterb[:, idx] + perturb_sign * scale * std
            else:
                X_purterb[:, idx] = 1 - X_purterb[:, idx]
            if f in [
                "CREAT_TX",
                "INR_TX",
                "TBILI_TX",
                "FINAL_SERUM_SODIUM",
                "DIAL_TX_1",
                "DIAL_TX_0",
            ]:
                MELD = meld(pd.DataFrame(X_purterb, columns=patient_features))
                MELD_old = meld(pd.DataFrame(X_te, columns=patient_features))
                mask1 = MELD != MELD_old
                MELD = meld_na(pd.DataFrame(X_purterb, columns=patient_features))
                MELD_old = meld_na(pd.DataFrame(X_te, columns=patient_features))
                mask2 = MELD != MELD_old
                mask = mask1 & mask2
                ([idx],) = np.where(patient_features == "MELD_PELD_LAB_SCORE")
                X_purterb[:, idx] = MELD
                X_purterb = X_purterb[mask]
                O_purterb = O_purterb[mask]
                X_test = X_test[mask]

        elif f in organ_features:
            ([idx],) = np.where(organ_features == f)
            if is_num:
                std = np.std(O_tr[:, idx])
                O_purterb[:, idx] = O_purterb[:, idx] + perturb_sign * scale * std
            else:
                O_purterb[:, idx] = 1 - O_purterb[:, idx]
        else:
            raise ValueError(f"feature {f} does not exist")

    criteria_purterb = create_criteria(
        pd.DataFrame(X_purterb, columns=patient_features),
        pd.DataFrame(O_purterb, columns=organ_features),
    )

    input_perturb = np.concatenate(
        [X_purterb, O_purterb, criteria_purterb.values], axis=1
    )
    return input_perturb, X_test


def single_feature_consistency(
    feature, X_train, X_test, data_dict, estimator, preprocessor
):
    consistency_masks = []
    for perturb_sign in [1.0, -1.0]:
        perturbed_features = [feature]
        perturb_signs = [perturb_sign]
        perturbed_samples, x_te = perturb_samples(
            X_train, X_test, perturbed_features, perturb_signs, data_dict, scale=0.25
        )
        if len(x_te) == 0:
            continue
        W, W0 = estimator.get_feature_importance(preprocessor.transform(x_te))
        prob_cf = estimator.get_counterfactual_predict(
            preprocessor.transform(perturbed_samples), W, W0
        )
        prob_base = estimator.predict_proba(preprocessor.transform(x_te))[:, 1]
        prob_perturb = estimator.predict_proba(
            preprocessor.transform(perturbed_samples)
        )[:, 1]
        expected_change = prob_cf - prob_base
        real_change = prob_perturb - prob_base
        consistency_mask = (real_change == 0) * (np.abs(expected_change) < 0.01) + (
            real_change != 0
        ) * (np.sign(real_change) == np.sign(expected_change))
        consistency_masks.append(consistency_mask)
    if len(consistency_masks) > 0:
        consistency_masks = np.concatenate(consistency_masks)
    if len(consistency_masks) < 50:
        consistency = np.nan
    else:
        consistency = np.sum(consistency_masks) / len(consistency_masks)
    return consistency


def single_feature_consistency_LIME(
    feature, X_train, X_test, data_dict, column_dict, estimator, preprocessor
):
    xo_dim = data_dict["x_dim"] + data_dict["o_dim"]
    c_dim = data_dict["c_dim"]
    categorical_features = np.concatenate(
        [column_dict["patient"][1], column_dict["organ"][1]]
    )
    LIME_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train[:, :xo_dim],
        categorical_features=categorical_features,
        class_names=["Decline", "Accept"],
        discretize_continuous=False,
        random_state=0,
    )

    predict_fn = lambda x: estimator.predict_proba(
        preprocessor.transform(np.concatenate([x, np.zeros((len(x), c_dim))], axis=1))
    )

    consistency_masks = []
    for perturb_sign in [1.0, -1.0]:
        perturbed_features = [feature]
        perturb_signs = [perturb_sign]
        perturbed_samples, x_te = perturb_samples(
            X_train, X_test, perturbed_features, perturb_signs, data_dict, scale=0.25
        )
        if len(x_te) == 0:
            continue
        prob_base = estimator.predict_proba(preprocessor.transform(x_te))[:, 1]
        prob_perturb = estimator.predict_proba(
            preprocessor.transform(perturbed_samples)
        )[:, 1]

        real_change = prob_perturb - prob_base
        expected_change = np.zeros_like(real_change)
        for i in range(len(perturbed_samples)):
            exp = LIME_explainer.explain_instance(
                perturbed_samples[i, :xo_dim],
                predict_fn,
                num_features=xo_dim,
                num_samples=50,
            )
            indices = [k for k, v in exp.as_map()[1]]
            weights = np.array([v for k, v in exp.as_map()[1]])
            prob_cf = np.sum(perturbed_samples[i, indices] * weights)
            expected_change[i] = prob_cf - prob_base[i]

        consistency_mask = (real_change == 0) * (np.abs(expected_change) < 0.01) + (
            real_change != 0
        ) * (np.sign(real_change) == np.sign(expected_change))
        consistency_masks.append(consistency_mask)
    if len(consistency_masks) > 0:
        consistency_masks = np.concatenate(consistency_masks)
    if len(consistency_masks) < 50:
        print(feature)
        consistency = np.nan
    else:
        consistency = np.sum(consistency_masks) / len(consistency_masks)
    return consistency


def consistency_evaluation(data_dir, center, report_dir, seed, max_iter=200):
    np.random.seed(seed)

    create_dir_if_not_exist(report_dir)
    data_source = aggregate_data(data_dir, [center])
    X, y, data_dict, column_dict = create_dataset(data_source, discretize=False)

    perturbed_features = [
        "DIAL_TX_1",
        "CREAT_TX",
        "INR_TX",
        "TBILI_TX",
        "FINAL_SERUM_SODIUM",
        "AGE_DON",
        "SGOT_DON",
        "SGPT_DON",
        "HGT_CM_DON_CALC",
        "WGT_KG_DON_CALC",
        "SHARE_TY_3",
        "SHARE_TY_4",
        "SHARE_TY_5",
        "HEP_C_ANTI_DON_1",
        "HBV_CORE_DON_1",
        "ETHCAT_DON_2",
        "GENDER_DON_0",
        "deathcirc_1",
        "NON_HRT_DON_1",
        "macro_1",
        "macro_2",
        "micro_1",
        "micro_2",
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    index = np.random.choice(X_test.shape[0], 200, replace=False)
    X_test = X_test[index]

    report_file = os.path.join(f"{report_dir}", "consistency.csv")

    considered_patient_features = [
        "DIAL_TX_1",
        "CREAT_TX",
        "INR_TX",
        "TBILI_TX",
        "FINAL_SERUM_SODIUM",
    ]
    considered_donor_features = [
        "AGE_DON",
        "SGOT_DON",
        "SGPT_DON",
        "HGT_CM_DON_CALC",
        "WGT_KG_DON_CALC",
        "SHARE_TY_3",
        "SHARE_TY_4",
        "SHARE_TY_5",
        "HEP_C_ANTI_DON_1",
        "HBV_CORE_DON_1",
        "ETHCAT_DON_2",
        "GENDER_DON_0",
        "deathcirc_1",
        "NON_HRT_DON_1",
        "macro_1",
        "macro_2",
        "micro_1",
        "micro_2",
    ]

    model_type = "iTransplant"
    hyper_parameter = {
        "input_space": "X",
        "criteria_space": "C",
        "h_dim": 30,
        "lambda_": 0.01,
        "num_experts": 20,
        "k": 8,
        "degree": 1,
        "num_layers": 1,
    }
    if os.path.exists(report_file):
        df = pd.read_csv(report_file, index_col=0)
        reports = [df]
    else:
        reports = []

    records = []
    for lambda_ in [0.0, 0.01, 0.1]:
        hyper_parameter["lambda_"] = lambda_

        clf, [preprocessor, estimator] = create_estimator(
            iTransplantEstimator,
            data_dict,
            column_dict,
            random_state=seed,
            hyper_parameter=hyper_parameter,
        )

        description = f"{estimator.name} with lambda={lambda_}"
        if len(reports) == 1 and description in reports[0]["model"].unique():
            continue

        clf.fit(
            X_train,
            y_train,
            estimator__validation_split=0.2,
            estimator__max_iter=max_iter,
            estimator__batch_size=400,
            estimator__verbose=True,
        )

        consistencies = []
        for feature in perturbed_features:
            consistency = single_feature_consistency(
                feature, X_train, X_test, data_dict, estimator, preprocessor
            )
            consistencies.append(consistency)
        record = [description] + consistencies
        records.append(record)

    df = pd.DataFrame(records, columns=["model"] + perturbed_features)
    mean_consistency = (
        df.loc[:, considered_patient_features].mean(axis=1) * 0.5
        + df.loc[:, considered_donor_features].mean(axis=1) * 0.5
    )
    df.loc[:, "avg consistency"] = mean_consistency

    reports.append(df)
    df = pd.concat(reports)
    df.to_csv(report_file)

    hyper_parameter = {
        "input_space": "C",
        "criteria_space": "C",
        "h_dim": 30,
        "lambda_": 0.01,
        "degree": 1,
        "num_layers": 2,
    }

    if os.path.exists(report_file):
        df = pd.read_csv(report_file, index_col=0)
        reports = [df]
    else:
        reports = []

    clf, [preprocessor, estimator] = create_estimator(
        INVASEEstimator,
        data_dict,
        column_dict,
        random_state=seed,
        hyper_parameter=hyper_parameter,
    )

    if len(reports) == 1 and estimator.name in reports[0]["model"].unique():
        pass
    else:
        clf.fit(
            X_train,
            y_train,
            estimator__validation_split=None,
            estimator__max_iter=max_iter,
            estimator__batch_size=400,
            estimator__verbose=True,
        )

        consistencies = []
        for feature in perturbed_features:
            consistency = single_feature_consistency_LIME(
                feature,
                X_train,
                X_test,
                data_dict,
                column_dict,
                estimator,
                preprocessor,
            )
            consistencies.append(consistency)
        record = [f"{estimator.name}"] + consistencies
        df = pd.DataFrame([record], columns=["model"] + perturbed_features)
        mean_consistency = (
            df.loc[:, considered_patient_features].mean(axis=1) * 0.5
            + df.loc[:, considered_donor_features].mean(axis=1) * 0.5
        )
        df.loc[:, "avg consistency"] = mean_consistency
        reports.append(df)
    df = pd.concat(reports)
    df.to_csv(report_file)

    hyper_parameter = {
        "input_space": "XxC",
        "criteria_space": "C",
        "h_dim": 50,
        "degree": 1,
        "num_layers": 4,
    }

    if os.path.exists(report_file):
        df = pd.read_csv(report_file, index_col=0)
        reports = [df]
    else:
        reports = []

    clf, [preprocessor, estimator] = create_estimator(
        BCEstimator,
        data_dict,
        column_dict,
        random_state=seed,
        hyper_parameter=hyper_parameter,
    )

    if len(reports) == 1 and estimator.name in reports[0]["model"].unique():
        pass
    else:
        clf.fit(
            X_train,
            y_train,
            estimator__validation_split=0.2,
            estimator__max_iter=max_iter,
            estimator__batch_size=400,
            estimator__verbose=True,
        )

        consistencies = []
        for feature in perturbed_features:
            consistency = single_feature_consistency_LIME(
                feature,
                X_train,
                X_test,
                data_dict,
                column_dict,
                estimator,
                preprocessor,
            )
            consistencies.append(consistency)
        record = [f"{estimator.name}"] + consistencies

        df = pd.DataFrame([record], columns=["model"] + perturbed_features)
        mean_consistency = (
            df.loc[:, considered_patient_features].mean(axis=1) * 0.5
            + df.loc[:, considered_donor_features].mean(axis=1) * 0.5
        )
        df.loc[:, "avg consistency"] = mean_consistency
        reports.append(df)
    df = pd.concat(reports)
    df.to_csv(report_file)
    return df.reset_index(drop=True)
