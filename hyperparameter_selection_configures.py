from sklearn.metrics import average_precision_score, roc_auc_score, make_scorer, log_loss

from src.models import iTransplantEstimator, INVASEEstimator, BCEstimator, DecisionTreeEstimator, LASSOEstimator, LogisticEstimator, LowessEstimator, RandomForestEstimator

data_dir = './data'

center = 'CTR20522'

max_iter = 200
cv_split = 5
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
    LogisticRegression_configure, iTransplant_configure, INVASE_configure,
    BC_configure, Lowess_configure
]
