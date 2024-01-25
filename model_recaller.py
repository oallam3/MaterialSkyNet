import pandas as pd
import pickle
import shap
from sklearn.preprocessing import StandardScaler
from src.models.evaluate_model import evaluate_model
from src.analysis.run_game import GameTheoryAnalyzer
from typing import List

class ModelRecaller:
    """
    A class for recalling, evaluating, and performing SHAP analysis on trained models.

    Attributes:
        model_name (str): Name of the model.
        bootstrap (str): Indicates if bootstrapping was used ('YES' or 'NO').
        data_file (str): Path to the data file.
        transform (str): Indicates if transformation is to be applied ('YES' or 'NO').
        n_models (int): Number of models to recall, relevant for bootstrapping.
        models (list): List to store recalled models.
        df_data (pd.DataFrame): Dataframe loaded from the data file.

    Methods:
        recall_and_evaluate(): Recalls and evaluates the models, optionally performing SHAP analysis.
        run_shap_analysis(model, X_train, X_test, y_train, y_test): Performs SHAP analysis on the given model and data.
    """

    def __init__(self, args: argparse.Namespace):
        self.model_name = args.model_name
        self.bootstrap = args.bootstrap
        self.data_file = args.data_file
        self.transform = args.transform
        self.n_models = args.n_models if args.bootstrap == 'YES' else 1
        self.models = []
        self.load_models()
        self.df_data = pd.read_csv(self.data_file)

    def load_models(self):
        """Loads the trained models into the models list."""
        if self.bootstrap == 'YES':
            for i in range(1, self.n_models + 1):
                model_path = f"models/{self.model_name}_bootstraps/{self.model_name}{i}.sav"
                self.models.append(pickle.load(open(model_path, 'rb')))
        else:
            model_path = f"models/{self.model_name}.sav"
            self.models.append(pickle.load(open(model_path, 'rb')))

    def recall_and_evaluate(self) -> List:
        """
        Recalls and evaluates the models, optionally performing SHAP analysis.

        Returns:
            List: A list of ModelResult objects containing evaluation metrics for each recalled model.
        """
        results = []
        for i, model in enumerate(self.models):
            print(f'Recalling Model No. {i + 1}/{len(self.models)}')
            X_train, X_test, y_train, y_test = self.split_data(i)
            model_result = evaluate_model(model, X_train, X_test, y_train, y_test, i + 1)
            results.append(model_result)
            if self.transform == 'YES':
                self.run_shap_analysis(model, X_train, X_test, y_train, y_test)
        return results

    def split_data(self, index) -> tuple:
        """
        Splits the data into training and testing sets based on the specified index.

        Args:
            index (int): Index for selecting the test set from stored indices.

        Returns:
            tuple: A tuple containing split training and testing data (X_train, X_test, y_train, y_test).
        """
        test_indices = pd.read_csv(self._get_indices_file_path(index))
        X_train = self.df_data.drop(test_indices[str(index + 1)].values.tolist()).drop(self.df_data.columns[-1], axis=1)
        y_train = pd.DataFrame(self.df_data.drop(test_indices[str(index + 1)].values.tolist())[self.df_data.columns[-1]], columns=[self.df_data.columns[-1]])
        X_test = self.df_data.iloc[test_indices[str(index + 1)].values.tolist()].drop(self.df_data.columns[-1], axis=1)
        y_test = pd.DataFrame(self.df_data.iloc[test_indices[str(index + 1)].values.tolist()][self.df_data.columns[-1]], columns=[self.df_data.columns[-1]])
        return X_train, X_test, y_train, y_test

    def run_shap_analysis(self, model, X_train, X_test, y_train, y_test):
        """
        Performs SHAP analysis on the given model and data.

        Args:
            model: The trained model for which SHAP analysis is to be performed.
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.DataFrame): Training labels.
            y_test (pd.DataFrame): Testing labels.
        """
        analyzer = GameTheoryAnalyzer(model, X_train, self.df_data.columns[:-1], transform=self.transform)
        shap_values = analyzer.run()
        # Save or plot SHAP values as needed

    def _get_indices_file_path(self, index) -> str:
        """
        Generates the file path for test indices based on the given index.

        Args:
            index (int): Index for selecting the test indices file.

        Returns:
            str: File path of the test indices file.
        """
        if self.bootstrap == 'YES':
            return f"test_indices_{self.model_name}_{self.n_models}_random_runs.csv"
        else:
            return f"test_indices_{self.model_name}.csv"
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recall and Evaluate Machine Learning Models.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to recall.")
    parser.add_argument("--bootstrap", type=str, choices=['YES', 'NO'], required=True, help="Indicate if bootstrapping was used.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the CSV data file.")
    parser.add_argument("--transform", type=str, choices=['YES', 'NO'], required=True, help="Indicate if data transformation is to be applied.")
    parser.add_argument("--n_models", type=int, default=1, help="Number of models to recall if bootstrapping was used.")

    args = parser.parse_args()
    recaller = ModelRecaller(args)
    results = recaller.recall_and_evaluate()
    # Process results as needed
