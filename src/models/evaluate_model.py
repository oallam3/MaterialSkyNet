import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
from typing import Optional, Dict, Any

class ModelResult:
    """
    A class to store and manage results of a machine learning model evaluation.

    Attributes:
        model: The evaluated machine learning model.
        train_mae (float): Mean Absolute Error on the training set.
        test_mae (float): Mean Absolute Error on the test set.
        train_mse (float): Mean Squared Error on the training set.
        test_mse (float): Mean Squared Error on the test set.
        train_r2 (float): R-squared value on the training set.
        test_r2 (float): R-squared value on the test set.
        model_number (int): Identifier for the model.
        feature_importances (Optional[np.ndarray]): Permutation importance of features.
        shap_values (Optional[shap.Explainer]): SHAP values for the model, if calculated.

    Methods:
        save_model(directory): Saves the model to the specified directory.
    """

    def __init__(self, model: Any, train_mae: float, test_mae: float, train_mse: float, test_mse: float, train_r2: float, test_r2: float, model_number: int, feature_importances: Optional[np.ndarray] = None, shap_values: Optional[shap.Explainer] = None):
        """
        Constructs all necessary attributes for the ModelResult object.

        Args:
            model: The evaluated machine learning model.
            train_mae (float): Mean Absolute Error on the training set.
            test_mae (float): Mean Absolute Error on the test set.
            train_mse (float): Mean Squared Error on the training set.
            test_mse (float): Mean Squared Error on the test set.
            train_r2 (float): R-squared value on the training set.
            test_r2 (float): R-squared value on the test set.
            model_number (int): Identifier for the model.
            feature_importances (Optional[np.ndarray], optional): Permutation importance of features. Defaults to None.
            shap_values (Optional[shap.Explainer], optional): SHAP values for the model, if calculated. Defaults to None.
        """
        self.model = model
        self.train_mae = train_mae
        self.test_mae = test_mae
        self.train_mse = train_mse
        self.test_mse = test_mse
        self.train_r2 = train_r2
        self.test_r2 = test_r2
        self.model_number = model_number
        self.feature_importances = feature_importances
        self.shap_values = shap_values

    def save_model(self, directory: str):
        """
        Saves the model to the specified directory.

        Args:
            directory (str): The directory path where the model will be saved.
        """
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"model_{self.model_number}.sav")
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

def evaluate_model(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, model_number: int, run_shap: bool = False) -> ModelResult:
    """
    Evaluates a machine learning model and returns its performance metrics.

    Args:
        model: The machine learning model to be evaluated.
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Testing feature set.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        model_number (int): Identifier for the model.
        run_shap (bool, optional): Flag to indicate if SHAP analysis should be performed. Defaults to False.

    Returns:
        ModelResult: An object containing evaluation metrics and other relevant details of the model.
    """
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    shap_values = None
    if run_shap:
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_train)
        
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    feature_importances = perm_importance.importances_mean

    return ModelResult(model, train_mae, test_mae, train_mse, test_mse, train_r2, test_r2, model_number, feature_importances, shap_values)
