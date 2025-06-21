from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def get_model(model_name, model_params=None):
    """
    Acts as a model factory. Returns an initialized model object based on its name.

    Args:
        model_name (str): The name of the model to retrieve (e.g., 'logistic_regression').
        model_params (dict, optional): A dictionary of parameters to pass to the model's constructor.

    Returns:
        An initialized scikit-learn model object.
    
    Raises:
        ValueError: If the model_name is not recognized.
    """
    model_params = model_params or {}

    if model_name == 'logistic_regression':
        # Set default params for logistic regression that can be overridden
        defaults = {'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1000, 'solver': 'liblinear'}
        params = {**defaults, **model_params}
        return LogisticRegression(**params)
    
    elif model_name == 'random_forest':
        # Set default params for random forest that can be overridden
        defaults = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        params = {**defaults, **model_params}
        return RandomForestClassifier(**params)

    elif model_name == 'lightgbm':
        # Set default params for LightGBM that can be overridden
        defaults = {'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        params = {**defaults, **model_params}
        return LGBMClassifier(**params)
    
    elif model_name == 'xgboost':
        # Set default params for XGBoost that can be overridden
        # scale_pos_weight is used to handle class imbalance, calculated from (count(neg)/count(pos))
        defaults = {'random_state': 42, 'n_jobs': -1, 'scale_pos_weight': 12}
        params = {**defaults, **model_params}
        return XGBClassifier(**params)

    elif model_name == 'catboost':
        # Set default params for CatBoost that can be overridden
        # scale_pos_weight is similar to XGBoost for handling class imbalance
        defaults = {'random_state': 42, 'thread_count': -1, 'scale_pos_weight': 12, 'verbose': 0}
        params = {**defaults, **model_params}
        return CatBoostClassifier(**params)

    # Add other models here in the future
    # elif model_name == 'svm':
    #     return SVC(**model_params)
    
    else:
        raise ValueError(f"Model '{model_name}' not recognized.") 