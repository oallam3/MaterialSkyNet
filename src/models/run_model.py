from model_factory import ModelFactory
from evaluate_model import evaluate_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from typing import List, Tuple, Optional

def run_model(model_type: str, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
              bootstrap: bool = False, n_models: int = 1, search_method: str = 'RS', hyperparams: Optional[dict] = None, 
              save_dir: str = 'models', run_shap: bool = False) -> Tuple[List, Optional[np.ndarray]]:
    """
    Trains and evaluates a machine learning model using specified hyperparameters and methods. Optionally performs SHAP analysis.

    Args:
        model_type (str): Type of the model to be trained (e.g., 'ANN', 'KRR').
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Testing data features.
        y_train (np.ndarray): Training data labels.
        y_test (np.ndarray): Testing data labels.
        bootstrap (bool, optional): Whether to use bootstrapping. Defaults to False.
        n_models (int, optional): Number of models to train if bootstrapping. Defaults to 1.
        search_method (str, optional): Method for hyperparameter tuning ('RS' for RandomizedSearchCV or 'GS' for GridSearchCV). Defaults to 'RS'.
        hyperparams (Optional[dict], optional): Hyperparameters for model tuning. Defaults to None.
        save_dir (str, optional): Directory to save the trained models. Defaults to 'models'.
        run_shap (bool, optional): Whether to perform SHAP analysis. Defaults to False.

    Returns:
        Tuple[List, Optional[np.ndarray]]: 
            - A list of ModelResult objects containing evaluation metrics for each model.
            - An aggregated SHAP values array if SHAP analysis is performed with bootstrapping, otherwise None.
    """

    model_factory = ModelFactory()
    tuner = HyperparameterTuner()

    model_results = []
    all_shap_values = []

    for i in range(n_models if bootstrap else 1):
        model = model_factory.get_model(model_type)
        tuned_model = tuner.tune(model, X_train, y_train, search_method, hyperparams)

        model_result = evaluate_model(tuned_model, X_train, X_test, y_train, y_test, model_number=i+1, run_shap=run_shap)
        model_result.save_model(save_dir)

        if run_shap:
            all_shap_values.append(model_result.shap_values)

        model_results.append(model_result)

    aggregated_shap_values = np.mean(all_shap_values, axis=0) if run_shap and bootstrap else None

    return model_results, aggregated_shap_values
