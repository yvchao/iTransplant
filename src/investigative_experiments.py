from pkg_resources import resource_filename
from sklearn.linear_model import LogisticRegression

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.cluster import KMeans

from sklearn import svm

from src.data_loading import aggregate_data, create_dataset
from src.utils import create_estimator, create_dir_if_not_exist
from src.models import iTransplantEstimator

import torch

torch.use_deterministic_algorithms(True)

import matplotlib

matplotlib.use('Agg')

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def prepare_data(data_dir, center, seed):
    data_source = aggregate_data(data_dir, [center])
    X, y, data_dict, column_dict = create_dataset(data_source,
                                                  discretize=False)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=seed,
                                                        stratify=y)

    return (X_train, y_train), (X_test, y_test), data_dict, column_dict


def get_saved_model_path(center, model_name, model_path="saved_models"):
    saved_models_path = resource_filename("src", model_path)
    create_dir_if_not_exist(saved_models_path)

    path_tail = os.path.join(model_path,
                             f'{center}-{model_name}-preprocessor.joblib')
    preprocessor_path = resource_filename("src", path_tail)
    model_name = f'{center}-{model_name}.pth'
    return model_name, preprocessor_path


def train_model(center, X_train, y_train, data_dict, column_dict, max_iter,
                seed, hyper_parameters):
    clf, [preprocessor,
          estimator] = create_estimator(iTransplantEstimator,
                                        data_dict,
                                        column_dict,
                                        random_state=seed,
                                        hyper_parameter=hyper_parameters)

    clf.fit(X_train,
            y_train,
            estimator__validation_split=0.2,
            estimator__max_iter=max_iter,
            estimator__batch_size=400,
            estimator__verbose=True)

    model_name, preprocessor_path = get_saved_model_path(
        center, estimator.name)
    joblib.dump(preprocessor, preprocessor_path)
    estimator.save_model(model_name)
    return estimator, preprocessor


def search_threshold(estimator,
                     preprocessor,
                     X_test,
                     y_test,
                     t_max=10.0,
                     epsilon=1e-2,
                     factor=0.05):
    X_scaled = preprocessor.transform(X_test)
    baseline = estimator.score(X_scaled, y_test, threshold=0)
    upper_bound = baseline - factor * baseline
    t0, t1 = 0, t_max
    delta_t0 = upper_bound - estimator.score(X_scaled, y_test, threshold=t0)
    delta_t1 = upper_bound - estimator.score(X_scaled, y_test, threshold=t1)
    assert delta_t0 <= 0
    assert delta_t1 > 0

    while np.abs(t1 - t0) >= epsilon:
        t_mid = 0.5 * (t0 + t1)
        delta_t_mid = upper_bound - estimator.score(
            X_scaled, y_test, threshold=t_mid)
        if delta_t_mid <= 0:
            t0 = t_mid
        else:
            t1 = t_mid
    return t0


def population_level_decision_drivers(center, X_test, y_test, data_dict,
                                      column_dict, hyper_parameters):
    clf, [preprocessor,
          estimator] = create_estimator(iTransplantEstimator,
                                        data_dict,
                                        column_dict,
                                        random_state=None,
                                        hyper_parameter=hyper_parameters)

    model_name, preprocessor_path = get_saved_model_path(
        center, estimator.name)
    preprocessor = joblib.load(preprocessor_path)
    estimator.load_model(model_name)

    threshold = search_threshold(estimator,
                                 preprocessor,
                                 X_test,
                                 y_test,
                                 t_max=1,
                                 factor=0.001)
    estimator._nn.threshold = 0

    X_scaled = preprocessor.transform(X_test)
    nn_output = estimator.expose_nn_output(X_scaled)
    W = nn_output['w']
    W_norm = W / (np.linalg.norm(W, axis=-1, keepdims=True) + 1e-10)

    label_criteria = data_dict['criteria']

    df = pd.DataFrame(data=W_norm, columns=label_criteria.to_list())
    fig, ax = plt.subplots(figsize=(10, 6))
    box = df.boxplot(column=list(reversed(label_criteria)),
                     showfliers=False,
                     ax=ax,
                     return_type='dict',
                     vert=False)
    xmin, xmax = ax.get_ylim()
    ax.plot([threshold, threshold], [xmin, xmax], 'b--', alpha=0.5, zorder=-1)
    ax.plot([-threshold, -threshold], [xmin, xmax],
            'b--',
            alpha=0.5,
            zorder=-1)
    ax.set_xlim([-0.9, 0.9])
    highlight_criteria = [
        'Donor Age', 'MELD-Na', 'MELD', 'Donor Weight',
        'National Donor', 'Local Donor', 'Regional Donor',
        'HCV Positive Donor', 'HBV Positive Donor',
        'Donation After Natural Death', 'Non-heart-beating Donation',
        'High Degree MaS'
    ]
    for criterion in highlight_criteria:
        [idx
         ], = np.where(criterion == np.array(list(reversed(label_criteria))))
        box['boxes'][idx].set_color('r')
        box['medians'][idx].set_color('r')
        box['whiskers'][2 * idx].set_color('r')
        box['whiskers'][2 * idx + 1].set_color('r')
        box['caps'][2 * idx].set_color('r')
        box['caps'][2 * idx + 1].set_color('r')

    fig.tight_layout()
    return fig


def plot_polar_density(X, labels, baseline=True, y_lims=None):
    assert len(X.shape) == 2 and len(labels) == X.shape[1]
    N_labels = len(labels)
    N = len(X)
    assert N > 0
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    if y_lims is not None:
        ax.set_ylim(y_lims)
    else:
        ax.set_ylim([-1.05, 1.05])

    mean = np.mean(X, axis=0)
    means = np.array(list(mean) + [mean[0]])

    angles = [i / float(N_labels) * 2 * np.pi for i in range(N_labels)]
    angles += [angles[0]]
    ax.plot(angles, means, '--')
    if baseline:
        theta = np.arange(0, 2, 1. / 180) * np.pi
        ax.plot(theta, np.zeros((len(theta), )), '--r', alpha=0.5)

    for i in np.arange(0.0, 0.6, 0.1):
        X_high = np.quantile(X, 0.5 + i, axis=0)
        X_low = np.quantile(X, 0.5 - i, axis=0)
        maxs = np.array(list(X_high) + [X_high[0]])
        mins = np.array(list(X_low) + [X_low[0]])

        ax.fill_between(angles, mins, maxs, alpha=0.1, color='grey')
    ax.set_theta_offset(np.pi * 72 / 180)

    ax.set_rlabel_position(75)
    ax.xaxis.set_tick_params(pad=15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, rotation=0)

    ax.grid(True)
    #ax.set_title('Weight Distribution Over Match Criteria')

    fig.tight_layout()
    return fig, ax


def find_best_cluster_num(weight, feature, n_range=[2, 10], random_state=0):
    n_min, n_max = n_range
    variance_cross_cluster = []
    n_list = np.arange(n_min, n_max + 1)
    for n in n_list:
        cls = KMeans(n_clusters=n, random_state=random_state)
        c_label = cls.fit_predict(feature)
        mean_weights_incluster = []
        for i in range(n):
            mask = c_label == i
            mean_weights_incluster.append(np.mean(weight[mask], axis=0))
        variance_cross_cluster.append(np.std(mean_weights_incluster))
    return n_list[np.argsort(variance_cross_cluster)]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_decision_boundary(X,
                           label,
                           w0,
                           w1,
                           titles=['Logistic Regression', 'iTransplant']):
    ssss = 2.5
    base_vector = np.vstack([w0, w1]).T
    q, r = np.linalg.qr(base_vector, mode='complete')
    q, r = np.linalg.qr(base_vector, mode='reduced')
    Y = X @ q
    y1s = np.linspace(-ssss, ssss, 30)
    y2s = np.linspace(-ssss, ssss, 30)
    y1mesh, y2mesh = np.meshgrid(y1s, y2s)
    w0_policy = np.vectorize(lambda y1, y2: sigmoid(y1 * r[0, 0]))
    w1_policy = np.vectorize(
        lambda y1, y2: sigmoid(y1 * r[0, 1] + y2 * r[1, 1]))

    cm = plt.cm.get_cmap('coolwarm')
    fig = plt.figure(figsize=(13, 6), constrained_layout=False)
    spec = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[10, 10, 0.5])

    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])

    clf = svm.LinearSVC(C=1.3, class_weight='balanced',
                        fit_intercept=True).fit(Y, label)
    w = clf.coef_[0]
    k = -w[0] / w[1]
    xx = np.linspace(-ssss, ssss)
    yy = k * xx - (clf.intercept_) / w[1]
    ax1.plot(xx,
             yy,
             '-.',
             linewidth=2,
             color='black',
             label='Optimal Decision Boundary')
    ax2.plot(xx,
             yy,
             '-.',
             linewidth=2,
             color='black',
             label='Optimal Decision Boundary')

    im1 = ax1.contourf(y1mesh,
                       y2mesh,
                       w0_policy(y1mesh, y2mesh),
                       cmap=cm,
                       levels=10)
    mask = label == 0
    ax1.scatter(Y[~mask, 0],
                Y[~mask, 1],
                c='cyan',
                marker='.',
                s=50,
                label='Accept',
                zorder=2)
    ax1.scatter(Y[mask, 0],
                Y[mask, 1],
                c='yellow',
                marker='x',
                label='Decline',
                zorder=1)

    ax1.legend()
    ax1.set_title(titles[0])

    im2 = ax2.contourf(y1mesh,
                       y2mesh,
                       w1_policy(y1mesh, y2mesh),
                       cmap=cm,
                       levels=10)
    mask = label == 0
    ax2.scatter(Y[~mask, 0],
                Y[~mask, 1],
                c='cyan',
                marker='.',
                s=50,
                label='Accept',
                zorder=2)
    ax2.scatter(Y[mask, 0],
                Y[mask, 1],
                c='yellow',
                marker='x',
                label='Decline',
                zorder=1)

    ax2.legend()
    ax2.set_title(titles[1])
    for ax in [ax1, ax2]:
        ax.set_xlim([-ssss, ssss])
        ax.set_ylim([-ssss, ssss])
    fig.colorbar(im2, cax=ax3)
    fig.tight_layout()
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.81, 0.08, 0.02, 0.88])
    #fig.colorbar(im2, cax=cbar_ax)
    return fig, [ax1, ax2]


def plot_decision_boundary_multi_clusters(W,
                                          X,
                                          c_label,
                                          clusters,
                                          a,
                                          n0,
                                          plot_logistic_regression=False):
    cm = plt.cm.get_cmap('coolwarm')

    fig, axs = plt.subplots(1, len(clusters), figsize=(16, 6))
    y1s = np.linspace(-2.5, 2.5, 30)
    y2s = np.linspace(-2.5, 2.5, 30)
    y1mesh, y2mesh = np.meshgrid(y1s, y2s)

    handles = []

    for i, c in enumerate(clusters):
        loc = c_label == c
        w1 = np.mean(W[loc], axis=0)
        base_vector = np.vstack([n0, w1]).T
        q, r = np.linalg.qr(base_vector, mode='complete')
        q, r = np.linalg.qr(base_vector, mode='reduced')
        Y = X[loc] @ q
        if plot_logistic_regression:
            w_policy = np.vectorize(lambda y1, y2: sigmoid(y1 * r[0, 0]))
        else:
            w_policy = np.vectorize(
                lambda y1, y2: sigmoid(y1 * r[0, 1] + y2 * r[1, 1]))

        c_value = a[loc]
        row = i // 3
        col = i % 3
        im = axs[col].contourf(y1mesh,
                               y2mesh,
                               w_policy(y1mesh, y2mesh),
                               cmap=cm,
                               levels=10)
        mask = c_value == 0
        l1 = axs[col].scatter(Y[~mask, 0],
                              Y[~mask, 1],
                              c='cyan',
                              marker='.',
                              s=50,
                              label='Acceptance',
                              zorder=2)
        l2 = axs[col].scatter(Y[mask, 0],
                              Y[mask, 1],
                              c='yellow',
                              marker='x',
                              label='Rejection',
                              zorder=1)
        axs[col].set_title(f'Cluster {c+1}')
        axs[col].axis('off')
        if not handles:
            handles = [l1, l2]

    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.04, 0.03, 0.75])
    fig.colorbar(im, cax=cbar_ax)
    fig.legend(handles, ['Accept', 'Decline'],
               bbox_to_anchor=(0.8, 0.95),
               loc='upper left')
    return fig, axs


def patient_specific_organ_preferences(center, X_train, y_train, X_test,
                                       y_test, data_dict, column_dict,
                                       hyper_parameters, seed):
    clf, [preprocessor,
          estimator] = create_estimator(iTransplantEstimator,
                                        data_dict,
                                        column_dict,
                                        random_state=seed,
                                        hyper_parameter=hyper_parameters)

    model_name, preprocessor_path = get_saved_model_path(
        center, estimator.name)
    preprocessor = joblib.load(preprocessor_path)
    estimator.load_model(model_name)

    X_scaled = preprocessor.transform(X_test)
    estimator._nn.threshold = 0
    nn_output = estimator.expose_nn_output(X_scaled)
    W = nn_output['w']
    Z = nn_output['z']
    gates = nn_output['gates']
    a_pred = nn_output['action']
    a = y_test
    W_norm = W / (np.linalg.norm(W, axis=-1, keepdims=True) + 1e-10)

    criteria = X_scaled[:, data_dict['X']['criteria']]

    label_criteria = data_dict['criteria']
    label_patient = data_dict['patient_features']

    LR = LogisticRegression(penalty='none',
                            random_state=seed,
                            solver='saga',
                            class_weight='balanced',
                            fit_intercept=False)
    LR_train = preprocessor.transform(X_train)[:, data_dict['X']['criteria']]
    LR.fit(LR_train, y_train)

    best_cluster_nums = find_best_cluster_num(W,
                                              Z,
                                              n_range=[2, 10],
                                              random_state=seed)

    n_clusters = best_cluster_nums[0]
    cls = KMeans(n_clusters=n_clusters, random_state=seed)
    c_label = cls.fit_predict(Z)

    a_pred_LR = LR.predict_proba(criteria)[:, 1]

    cluster_wise_apr = []
    for c in range(n_clusters):
        mask = c_label == c

        cluster_label = f'cluster {c+1}'

        auc_prc = average_precision_score(a[mask], a_pred[mask])
        auc_prc_LR = average_precision_score(a[mask], a_pred_LR[mask])
        in_cluster_apr = {
            'cluster': cluster_label,
            'sample number': np.sum(mask),
            f'AUC-PRC ({estimator.name})': auc_prc,
            'AUC-PRC (Logistic Regression)': auc_prc_LR
        }
        cluster_wise_apr.append(in_cluster_apr)
    cluster_wise_apr = pd.DataFrame(cluster_wise_apr).set_index('cluster')

    cluster_wise_policy_signature_dist = {}
    cluster_wise_patient_features_dist = []
    for c in range(n_clusters):
        mask = c_label == c
        cluster_label = f'cluster {c+1}'
        fig, ax = plot_polar_density(W_norm[mask] - np.mean(W_norm, axis=0),
                                     label_criteria.to_list(),
                                     baseline=True,
                                     y_lims=[-0.65, 0.49])
        cluster_wise_policy_signature_dist[cluster_label] = fig

        patient_features_dist = {}
        patient_features_dist['cluster'] = cluster_label
        patient_features = X_test[:, data_dict['X']['patient']]
        [idx_hcc0], = np.where(label_patient == 'statushcc_0')
        patient_features_dist['HCC negative (%)'] = 100 * patient_features[
            mask, idx_hcc0].sum() / mask.sum()

        [idx_hcc1], = np.where(label_patient == 'statushcc_1')
        patient_features_dist['HCC positive (%)'] = 100 * patient_features[
            mask, idx_hcc1].sum() / mask.sum()

        [idx_meld], = np.where(label_patient == 'MELD_PELD_LAB_SCORE')
        patient_features_dist['HCC positive & MELD <= 20 (%)'] = 100 * (
            (patient_features[mask, idx_hcc1]) *
            (patient_features[mask, idx_meld] <= 20)).sum() / mask.sum()

        for i in range(0, 40, 10):
            patient_features_dist[f'{i} <= MELD < {i+10} (%)'] = 100 * (
                (patient_features[mask, idx_meld] >= i) &
                (patient_features[mask, idx_meld] < i + 10)
            ).sum() / mask.sum()

        patient_features_dist['MELD >= 40 (%)'] = 100 * (
            (patient_features[mask, idx_meld] >= 40)).sum() / mask.sum()
        cluster_wise_patient_features_dist.append(patient_features_dist)
    cluster_wise_patient_features_dist = pd.DataFrame(
        cluster_wise_patient_features_dist).set_index('cluster')

    ret_main = {}
    ret_appendix = {}
    ret_appendix['cluster-wise apr improvement'] = cluster_wise_apr
    ret_main[
        'cluster-wise policy signature distribution'] = cluster_wise_policy_signature_dist
    ret_main[
        'cluster-wise patient feature distribution'] = cluster_wise_patient_features_dist

    cluster_wise_expert_selection = {}
    for c in range(n_clusters):
        mask = c_label == c
        df = pd.DataFrame(data=gates[mask],
                          columns=np.arange(1, 1 + estimator.num_experts))
        fig, ax = plt.subplots(figsize=(8, 6))
        box = df.boxplot(showfliers=False, ax=ax, return_type='dict')
        ax.set_ylim([-0.1, 0.9])
        ax.set_ylabel('Selection Probability')
        ax.set_xlabel('Indices of expert networks')
        fig.tight_layout()
        cluster_wise_expert_selection[f'cluster {c+1}'] = fig

    ret_appendix[
        'cluster-wise expert selection'] = cluster_wise_expert_selection

    considered_clusters = []
    for i in range(n_clusters):
        mask = c_label == i
        if np.sum(y_test[mask]) < 30:
            continue
        considered_clusters.append(i)

    n0 = LR.coef_[0]

    cluster_wise_decision_boundary_comparison_with_LR = {}
    for c in considered_clusters:
        loc = c_label == c
        n1 = np.mean(W[loc], axis=0)
        fig, ax = plot_decision_boundary(criteria[loc], a[loc], n0, n1)
        cluster_wise_decision_boundary_comparison_with_LR[
            f'cluster {c+1}'] = fig

    ret_appendix[
        'decision boundary comparison with LR'] = cluster_wise_decision_boundary_comparison_with_LR

    fig, ax = plot_decision_boundary_multi_clusters(W, criteria, c_label,
                                                    considered_clusters[:3], a,
                                                    n0)
    ret_appendix['decision boundaries'] = fig
    return ret_main, ret_appendix


def cross_center_variation(center_A,
                           center_B,
                           X_test,
                           y_test,
                           data_dict,
                           column_dict,
                           hyper_parameters,
                           seed=None,
                           patient_idx=None):
    _, [preprocessor,
        estimator] = create_estimator(iTransplantEstimator,
                                      data_dict,
                                      column_dict,
                                      random_state=seed,
                                      hyper_parameter=hyper_parameters)

    model_name, preprocessor_path = get_saved_model_path(
        center_A, estimator.name)
    preprocessor = joblib.load(preprocessor_path)
    estimator.load_model(model_name)

    _, [preprocessor1,
        estimator1] = create_estimator(iTransplantEstimator,
                                       data_dict,
                                       column_dict,
                                       random_state=seed,
                                       hyper_parameter=hyper_parameters)

    model_name, preprocessor_path = get_saved_model_path(
        center_B, estimator1.name)
    preprocessor1 = joblib.load(preprocessor_path)
    estimator1.load_model(model_name)

    nn_output = estimator.expose_nn_output(preprocessor.transform(X_test))
    a_pred = nn_output['action'][:, 0]

    nn_output = estimator1.expose_nn_output(preprocessor1.transform(X_test))
    a_pred1 = nn_output['action'][:, 0]

    a = y_test

    ret = {}
    df = pd.DataFrame()
    df.loc[
        0,
        'difference in average organ offer acceptance rate'] = f'{np.abs(a_pred.mean()-a_pred1.mean()):.3f}'
    df.loc[
        0,
        'deviations of learned policies at individual level'] = f'{np.abs(a_pred-a_pred1).mean():.3f}±{np.abs(a_pred-a_pred1).std():.3f}'
    ret['Population level deviations of learned policies'] = df

    label_criteria = data_dict['criteria']
    label_patient = data_dict['patient_features']
    label_organ = data_dict['organ_features']

    raw_patient_features = X_test[:, data_dict['X']['patient']]
    raw_organ_features = X_test[:, data_dict['X']['organ']]
    raw_criteria = X_test[:, data_dict['X']['criteria']]

    label2idx_p = {
        'AGE': 0,
        'MELD_PELD_LAB_SCORE': 7,
        'INR_TX': 5,
        'CREAT_TX': 4,
        'TBILI_TX': 6
    }
    label2idx_m = {'SGOT_DON': 5, 'AGE_DON': 2}
    label2idx_o = {'SGOT_DON': 5, 'AGE_DON': 0}

    candicates_indices, = np.where(
        (np.abs(a_pred - a_pred1) > 0.3) * (a == 1) *
        (raw_patient_features[:, label2idx_p['MELD_PELD_LAB_SCORE']] > 20) *
        (raw_patient_features[:, label2idx_p['AGE']] < 60) *
        (raw_organ_features[:, label2idx_o['SGOT_DON']] > 1) *
        (raw_organ_features[:, label2idx_o['SGOT_DON']] < 3000))

    ret['cases where deviations arise'] = candicates_indices
    if not patient_idx:
        idx = candicates_indices[0]
    else:
        idx = patient_idx
    df = pd.DataFrame(columns=label_patient)
    df.loc[0] = raw_patient_features[idx]
    ret['patient features'] = df

    df = pd.DataFrame(columns=label_organ)
    df.loc[0] = raw_organ_features[idx]
    ret['donor features'] = df

    ret['P_A'] = a_pred[idx]
    ret['P_B'] = a_pred1[idx]
    ret['action'] = a[idx]

    patient_feature = raw_patient_features[idx]
    organ_feature = raw_organ_features[idx]
    criteria = raw_criteria[idx]

    pred = a_pred[idx]
    pred1 = a_pred1[idx]

    def p_accept(AGE_DON=None, SGOT_DON=None):
        X = patient_feature.copy()
        O = organ_feature.copy()
        C = criteria.copy()
        if AGE_DON is not None:
            C[label2idx_m['AGE_DON']] = AGE_DON
        if SGOT_DON is not None:
            C[label2idx_m['SGOT_DON']] = SGOT_DON
        fake_sample = np.concatenate([X, O, C])[np.newaxis, :]
        p = estimator.predict_proba(preprocessor.transform(fake_sample))[0, 1]
        return p.astype('float64')

    def p_accept1(AGE_DON=None, SGOT_DON=None):
        X = patient_feature.copy()
        O = organ_feature.copy()
        C = criteria.copy()
        if AGE_DON is not None:
            C[label2idx_m['AGE_DON']] = AGE_DON
        if SGOT_DON is not None:
            C[label2idx_m['SGOT_DON']] = SGOT_DON
        fake_sample = np.concatenate([X, O, C])[np.newaxis, :]
        p = estimator1.predict_proba(preprocessor1.transform(fake_sample))[0,
                                                                           1]
        return p.astype('float64')

    f = np.vectorize(p_accept)
    f1 = np.vectorize(p_accept1)

    fig, ax = plt.subplots(figsize=(6, 4))
    SGOT_DON = np.linspace(0, 2000)
    real_sgot_don = organ_feature[label2idx_o['SGOT_DON']]
    ax.plot(SGOT_DON, f(SGOT_DON=SGOT_DON), color='blue', label='Center A')
    ax.scatter(real_sgot_don, pred, color='blue', marker='o', s=50)
    ax.plot([real_sgot_don, real_sgot_don], [0, pred], '--')

    ax.plot(SGOT_DON,
            f1(SGOT_DON=SGOT_DON),
            '-.',
            color='green',
            label='Center B')
    ax.scatter(real_sgot_don, pred1, color='green', marker='o', s=50)

    ax.set_xlabel('Donor AST')
    ax.set_ylabel('P(Acceptance)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0.0, 1.01)
    fig.tight_layout()
    ret['Counter factual donor AST'] = fig

    fig, ax = plt.subplots(figsize=(6, 4))
    AGE_DON = np.linspace(0, 80)
    ax.plot(AGE_DON, f(AGE_DON=AGE_DON), color='blue', label='Center A')
    real_age_don = organ_feature[label2idx_o['AGE_DON']]
    ax.scatter(real_age_don, pred, color='blue', marker='o', s=50)
    ax.plot([real_age_don, real_age_don], [0, pred], '--')

    ax.plot(AGE_DON,
            f1(AGE_DON=AGE_DON),
            '-.',
            color='green',
            label='Center B')
    ax.scatter(real_age_don, pred1, color='green', marker='o', s=50)
    ax.set_xlabel('Donor Age')
    ax.set_ylabel('P(Acceptance)')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    ret['Counter factual donor age'] = fig

    return ret
