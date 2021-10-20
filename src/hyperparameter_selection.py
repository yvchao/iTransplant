from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score,roc_auc_score,make_scorer,log_loss

import os
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,make_scorer,log_loss
import tqdm

from src.data_loading import aggregate_data, create_dataset
from src.utils import create_estimator, create_dir_if_not_exist
from src.models import iTransplantEstimator, INVASEEstimator, BCEstimator, DecisionTreeEstimator, LASSOEstimator,LogisticEstimator,LowessEstimator,RandomForestEstimator

import torch
import numpy as np

scoring = {'AUC-ROC': make_scorer(roc_auc_score,needs_proba=True,),
           'AUC-PRC': make_scorer(average_precision_score,needs_proba=True),
           'NLL':make_scorer(lambda y_ture,y_pred,sample_weight=None:log_loss(y_ture,y_pred,eps=1e-7,sample_weight=sample_weight),greater_is_better=False,needs_proba=True)}

parameter_ranges_table={
    'iTransplant': { 'num_layers':[1,2,3], 'h_dim':[30,40,50], 'lambda_': [0.01,0.05], 'num_experts': [10,20], 'k': [4,8]},
    'INVASE': {'num_layers':[2,4,6], 'h_dim': [20,30], 'lambda_':[0.01,0.1,0.5]},
    'BC': {'h_dim':[10,20,30,40,50],'num_layers':[2,4,6,8]},
    'LASSO': {'alpha': [0.01,0.1,1.0,5.0]},
    'Random Forest': {'n_estimators': [10,50,100,150],'max_depth': [5,10,15]},
    'Logistic Regression': {'n_clusters': [1,2,4,8]},
    'Decision Tree': {'n_clusters': [1,2,4,8],'max_depth': [5,10,15]},
    'LOWESS': {'tau': [0.01,0.1,1,10]}
}

def hyperparameter_search(data_dir, report_dir, center, configurations, cv_split=5, max_iter=200, seed=19260817, verbose=0):
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    create_dir_if_not_exist(report_dir)
    # Aggregate data
    aggregate_data_source=aggregate_data(data_dir, [center])
    X,y,data_dict,column_dict=create_dataset(aggregate_data_source)

    tbar = tqdm.tqdm(configurations,position=0, leave=True)
    for i, configuration in enumerate(tbar):
        model = configuration['model']
        hyper_parameters = configuration['hyper_parameters']
        validation_split = configuration['validation_split']
        description = configuration['description']

        
        hyperparameter_selection_result = os.path.join(report_dir,f'{description}-hyperparameter-tune.csv')
        if os.path.exists(hyperparameter_selection_result):
            tbar.update(i)
            continue

        tbar.set_description(f'searching hyperparameters for {description} ...')

        clf,[preprocessor,estimator]=create_estimator(model,data_dict,column_dict,random_state=seed,hyper_parameter=hyper_parameters)

        parameter_ranges = parameter_ranges_table[estimator.name]

        param_grid={f'estimator__{k}':v for k,v in parameter_ranges.items()}
        
        GS=GridSearchCV(clf, param_grid, scoring=scoring,refit=False,cv=cv_split,verbose=verbose)

        _=GS.fit(X,y, estimator__validation_split=validation_split, estimator__max_iter = max_iter, estimator__verbose=False)

        df=pd.DataFrame(GS.cv_results_)

        df.to_csv(hyperparameter_selection_result)
    tbar.close()

def read_optimal_hyperparameter(description, report_dir, metric='NLL'):
    hyperparameter_selection_result = os.path.join(report_dir,f'{description}-hyperparameter-tune.csv')

    df=pd.read_csv(hyperparameter_selection_result)

    df = df.sort_values(f'rank_test_{metric}')
    params = df.loc[:, df.columns.str.startswith('param_estimator__')]
    scores = df.loc[:, [f'mean_test_{metric}', f'rank_test_{metric}']]
    df = pd.concat([params, scores],axis=1)

    df.columns = df.columns.str.replace(r'^param_estimator__', '', regex=True)
    return df.head(2).set_index(f'rank_test_{metric}')
