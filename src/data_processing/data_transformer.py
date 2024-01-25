import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from typing import Tuple

class DataTransformer:
    """
    A class for applying various data transformations to feature sets.

    Attributes:
        compound (bool): Whether to generate compound features (feature interactions).
        bootstrap (bool): Whether the transformer is used in a bootstrapping context.
    
    Methods:
        transform(X_train, X_test, feature_cols): Transforms the training and test sets.
    """

    def __init__(self, compound: bool = False, bootstrap: bool = False):
        """
        Constructs all necessary attributes for the DataTransformer object.

        Args:
            compound (bool, optional): Flag to create compound features. Defaults to False.
            bootstrap (bool, optional): Flag to indicate bootstrap context. Defaults to False.
        """
        self.compound = compound
        self.bootstrap = bootstrap

    def transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
        """
        Applies scaling and transformations to the training and test sets.

        Args:
            X_train (pd.DataFrame): Training feature set.
            X_test (pd.DataFrame): Test feature set.
            feature_cols (list): List of feature column names.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, list]: Transformed training set, test set, and list of new feature columns.
        """
        # First-level scaling using StandardScaler
        X_train_scaled, X_test_scaled = self._scale_data(StandardScaler(), X_train, X_test)

        # Second-level scaling using MaxAbsScaler
        X_train_scaled, X_test_scaled = self._scale_data(MaxAbsScaler(), X_train_scaled, X_test_scaled)

        # Converting scaled data back to DataFrame
        X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

        # Applying transformations: square, sqrt, log, and exponential
        X_train_transformed = self._apply_transformations(X_train_df)
        X_test_transformed = self._apply_transformations(X_test_df)

        # Handling compound features if required
        if self.compound:
            X_train_transformed = self._create_compound_features(X_train_transformed)
            X_test_transformed = self._create_compound_features(X_test_transformed)

        # Save transformed data if not in bootstrap mode
        if not self.bootstrap:
            X_train_transformed.to_csv("res/X_train_trans_comp.csv")
            X_test_transformed.to_csv("res/X_test_trans_comp.csv")

        return X_train_transformed, X_test_transformed, X_train_transformed.columns

    def _scale_data(self, scaler: object, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scales the data using the provided scaler.

        Args:
            scaler (object): An instance of a Scikit-learn scaler.
            X_train (np.ndarray): Training data to scale.
            X_test (np.ndarray): Test data to scale.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scaled training and test data.
        """
        scaler.fit(X_train)
        return scaler.transform(X_train), scaler.transform(X_test)

    def _apply_transformations(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies various mathematical transformations to each feature.

        Args:
            X_df (pd.DataFrame): Dataframe containing the features.

        Returns:
            pd.DataFrame: Transformed features.
        """
        for feature in X_df.columns:
            X_df[feature + '^2'] = X_df[feature] ** 2
            X_df['(' + feature + '+1)^1/2'] = (X_df[feature] + 1) ** 0.5
            X_df['log(' + feature + '+2)'] = np.log10(X_df[feature] + 2)
            X_df['e^(' + feature + ')'] = np.exp(X_df[feature])

        # Handle any NaN values that might have been introduced
        X_df.fillna(X_df.mean(), inplace=True)

        return X_df

    def _create_compound_features(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates compound features by multiplying pairs of existing features.

        Args:
            X_df (pd.DataFrame): Dataframe with existing features.

        Returns:
            pd.DataFrame: Dataframe with added compound features.
        """
        feature_cols = X_df.columns.tolist()
        for i, feature_i in enumerate(feature_cols):
            for feature_j in feature_cols[i + 1:]:
                compound_feature_name = feature_i + '*' + feature_j
                X_df[compound_feature_name] = X_df[feature_i] * X_df[feature_j]
        return X_df
