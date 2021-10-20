from sklearn.metrics import average_precision_score, roc_auc_score, make_scorer, log_loss

from src.models import iTransplantEstimator, INVASEEstimator, BCEstimator, DecisionTreeEstimator, LASSOEstimator, LogisticEstimator, LowessEstimator, RandomForestEstimator

data_dir = './data'

centers = [
    'CTR20522', 'CTR19034', 'CTR6820', 'CTR7471', 'CTR23312', 'CTR14942',
    'CTR16864', 'CTR23808', 'CTR25110', 'CTR13609'
]

scoring = {
    'AUC-ROC':
    make_scorer(roc_auc_score, needs_proba=True),
    'AUC-PRC':
    make_scorer(average_precision_score, needs_proba=True),
    'NLL':
    make_scorer(lambda y_ture, y_pred: log_loss(y_ture, y_pred, eps=1e-7),
                greater_is_better=False,
                needs_proba=True)
}

max_iter = 200
n_tests = 5
seed = 19260817

LASSO_configure = {
    'model': LASSOEstimator,
    'hyper_parameters': {
        'input_space': 'C',
        'criteria_space': 'C',
        'alpha': 0.01,
        'degree': 1
    },
    'validation_split': None,
    'description': 'LASSO'
}

RandomForest_configure = {
    'model': RandomForestEstimator,
    'hyper_parameters': {
        'input_space': 'C',
        'criteria_space': 'C',
        'n_estimators': 150,
        'max_depth': 15,
        'degree': 1
    },
    'validation_split': None,
    'description': 'Random Forest'
}

DecisionTree_configure = {
    'model': DecisionTreeEstimator,
    'hyper_parameters': {
        'input_space': 'X',
        'criteria_space': 'C',
        'max_depth': 5,
        'n_clusters': 1,
        'degree': 1
    },
    'validation_split': None,
    'description': 'Decision Tree'
}

PerClusterDecisionTree_configure = {
    'model': DecisionTreeEstimator,
    'hyper_parameters': {
        'input_space': 'X',
        'criteria_space': 'C',
        'max_depth': 5,
        'n_clusters': 2,
        'degree': 1
    },
    'validation_split': None,
    'description': 'Per-cluster Decision Tree'
}

LogisticRegression_configure = {
    'model': LogisticEstimator,
    'hyper_parameters': {
        'input_space': 'X',
        'criteria_space': 'C',
        'n_clusters': 1,
        'degree': 1
    },
    'validation_split': None,
    'description': 'Logistic Regression'
}

PerClusterLogisticRegression_configure = {
    'model': LogisticEstimator,
    'hyper_parameters': {
        'input_space': 'X',
        'criteria_space': 'C',
        'n_clusters': 2,
        'degree': 1
    },
    'validation_split': None,
    'description': 'Per-cluster Logistic Regression'
}

iTransplant_configure = {
    'model': iTransplantEstimator,
    'hyper_parameters': {
        'input_space': 'X',
        'criteria_space': 'C',
        'h_dim': 30,
        'lambda_': 0.01,
        'num_experts': 20,
        'k': 8,
        'degree': 1,
        'num_layers': 1
    },
    'validation_split': 0.2,
    'description': 'iTransplant'
}

INVASE_configure = {
    'model': INVASEEstimator,
    'hyper_parameters': {
        'input_space': 'C',
        'criteria_space': 'C',
        'h_dim': 30,
        'lambda_': 0.01,
        'degree': 1,
        'num_layers': 2
    },
    'validation_split': None,
    'description': 'INVASE'
}

BC_configure = {
    'model': BCEstimator,
    'hyper_parameters': {
        'input_space': 'XxC',
        'criteria_space': 'C',
        'h_dim': 50,
        'degree': 1,
        'num_layers': 4
    },
    'validation_split': 0.2,
    'description': 'BC'
}

Lowess_configure = {
    'model': LowessEstimator,
    'hyper_parameters': {
        'input_space': 'X',
        'criteria_space': 'C',
        'tau': 1,
        'degree': 1
    },
    'validation_split': None,
    'description': 'LOWESS'
}

configurations = [
    LASSO_configure, RandomForest_configure, DecisionTree_configure,
    PerClusterDecisionTree_configure, LogisticRegression_configure,
    PerClusterLogisticRegression_configure, iTransplant_configure,
    INVASE_configure, BC_configure, Lowess_configure
]
