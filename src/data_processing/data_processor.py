import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from typing import Tuple, List

class DataProcessor:
    """Processes data for machine learning models.

    Attributes:
        data_file (str): Path to the CSV data file.
        test_size (float): Proportion of the dataset to include in the test split.
        stratify (bool): Whether to stratify the split according to the target variable.
        transform_features (bool): Whether to apply transformations to features.
        compound (bool): Whether to create compound features.
        bootstrap (bool): Whether bootstrap aggregation is being used.

    """

    def __init__(self, data_file: str, test_size: float = 0.1, stratify: bool = None,
                 transform_features: bool = False, compound: bool = False, bootstrap: bool = False) -> None:
        self.data_file = data_file
        self.test_size = test_size
        self.stratify = stratify
        self.transform_features = transform_features
        self.compound = compound
        self.bootstrap = bootstrap

    def load_data(self) -> pd.DataFrame:
        """Loads data from the specified CSV file.

        Returns:
            DataFrame: The loaded data.

        """
        return pd.read_csv(self.data_file)

    def stratifier(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits data into training and test sets with stratification.

        Returns:
            Tuple[DataFrame, DataFrame, Series, Series]: Train features, test features, train target, and test target.

        """
        data = self.load_data()
        bin_count = 5
        bin_numbers = pd.qcut(data[data.columns[-1]], q=bin_count, labels=False, duplicates='drop')
        return train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=self.test_size, random_state=42, stratify=bin_numbers)

    def randomSeed_stratifier(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits data with random seed stratification.

        Returns:
            Tuple[DataFrame, DataFrame, Series, Series]: Train features, test features, train target, and test target.

        """
        data = self.load_data()
        bin_count = 5
        bin_numbers = pd.qcut(data[data.columns[-1]], q=bin_count, labels=False, duplicates='drop')
        return train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=self.test_size, stratify=bin_numbers)

    def data_split(self, i: int, indices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], pd.DataFrame]:
        """Splits and optionally transforms data.

        Args:
            i (int): Index for tracking bootstrap iterations.
            indices (DataFrame): DataFrame to store indices of test data.

        Returns:
            Tuple[DataFrame, DataFrame, Series, Series, List[str], DataFrame]: Train features, test features, train target, test target, feature columns, and updated indices.

        """
        if self.stratify:
            [X_train, X_test, y_train, y_test] = self.stratifier()
        else:
            data = self.load_data()
            X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=self.test_size, random_state=42)
        indices[i + 1] = X_test.index.values.tolist()
        
        feature_cols = X_train.columns[:]
        if self.transform_features:
            X_train, X_test, feature_cols = self.transform(X_train, X_test, feature_cols)
        else:
            X_train, X_test = self.standardize_data(X_train, X_test)
        return X_train, X_test, y_train, y_test, feature_cols, indices

    def standardize_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Standardizes the data using StandardScaler.

        Args:
            X_train (DataFrame): Training features.
            X_test (DataFrame): Testing features.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Standardized training and testing features.

        """
        scaler = StandardScaler()  
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
