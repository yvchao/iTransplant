from sklearn.metrics import average_precision_score, roc_auc_score, make_scorer, log_loss

from src.models import iTransplantEstimator, INVASEEstimator, BCEstimator, DecisionTreeEstimator, LASSOEstimator, LogisticEstimator, LowessEstimator, RandomForestEstimator

data_dir = './data'

center = 'CTR20522'

max_iter = 200
cv_split = 5
seed = 19260817

LASSO_XxC_configure = {
    'model': LASSOEstimator,
    'hyper_parameters': {
        'input_space': 'XxC',
        'criteria_space': 'C',
        'alpha': 0.01,
        'degree': 1
    },
    'validation_split': None,
    'description': 'LASSO (XxC)'
}

LASSO_XxO_configure = {
    'model': LASSOEstimator,
    'hyper_parameters': {
        'input_space': 'XxO',
        'criteria_space': 'C',
        'alpha': 0.01,
        'degree': 1
    },
    'validation_split': None,
    'description': 'LASSO (XxO)'
}

LogisticRegression_configure = {
    'model': LogisticEstimator,
    'hyper_parameters': {
        'input_space': 'X',
        'criteria_space': 'C',
        'n_clusters': 1,
        'degree': 2
    },
    'validation_split': None,
    'description': 'Logistic Regression (with interaction terms)'
}

INVASE_XxC_configure = {
    'model': INVASEEstimator,
    'hyper_parameters': {
        'input_space': 'XxC',
        'criteria_space': 'C',
        'h_dim': 30,
        'lambda_': 0.01,
        'degree': 1,
        'num_layers': 2
    },
    'validation_split': None,
    'description': 'INVASE (XxC)'
}

INVASE_XxO_configure = {
    'model': INVASEEstimator,
    'hyper_parameters': {
        'input_space': 'XxO',
        'criteria_space': 'C',
        'h_dim': 30,
        'lambda_': 0.01,
        'degree': 1,
        'num_layers': 2
    },
    'validation_split': None,
    'description': 'INVASE (XxO)'
}

BC_C_configure = {
    'model': BCEstimator,
    'hyper_parameters': {
        'input_space': 'C',
        'criteria_space': 'C',
        'h_dim': 50,
        'degree': 1,
        'num_layers': 4
    },
    'validation_split': 0.2,
    'description': 'BC (C)'
}

BC_XxO_configure = {
    'model': BCEstimator,
    'hyper_parameters': {
        'input_space': 'XxO',
        'criteria_space': 'C',
        'h_dim': 50,
        'degree': 1,
        'num_layers': 4
    },
    'validation_split': 0.2,
    'description': 'BC (XxO)'
}

iTransplant_XxO_configure = {
    'model': iTransplantEstimator,
    'hyper_parameters': {
        'input_space': 'X',
        'criteria_space': 'O',
        'h_dim': 30,
        'lambda_': 0.01,
        'num_experts': 20,
        'k': 8,
        'degree': 1,
        'num_layers': 1
    },
    'validation_split': 0.2,
    'description': 'iTransplant (XxO)'
}

configurations = [
    LASSO_XxC_configure, LASSO_XxO_configure, LogisticRegression_configure, INVASE_XxC_configure, INVASE_XxO_configure, BC_C_configure, 
    BC_XxO_configure, iTransplant_XxO_configure
]
