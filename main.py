import argparse
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from src.data_processing.data_processor import process_data  # Assuming this is a function in data_processor.py
from src.data_processing.data_transformer import DataTransformer
from src.models.run_model import run_model  # This would be your model training logic

def main(args: argparse.Namespace) -> None:
    """
    Main function to load data, process, transform, train models, and save the trained models.

    Args:
        args (argparse.Namespace): Command-line arguments passed to the script.
    """
    # Load data
    data = pd.read_csv(args.data_file)

    # Data preprocessing
    data = process_data(data)

    # Split data
    X = data.drop(args.target_column, axis=1)  # Replace 'target_column' with your actual target column name
    y = data[args.target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.testsize, random_state=42)

    # Data transformation (if required)
    if args.transform_features:
        transformer = DataTransformer(compound=args.compound, bootstrap=args.bootstrap)
        X_train, X_test, _ = transformer.transform(X_train, X_test, X.columns)

    # Model training
    models, results = run_model(args.model, X_train, X_test, y_train, y_test, bootstrap=args.bootstrap, n_models=args.n_models)

    # Save the models
    for i, model in enumerate(models):
        with open(f'models/{args.model}_{i+1}.sav', 'wb') as f:
            pickle.dump(model, f)

    # Print results
    print("Model Training Complete. Results:")
    for result in results:
        print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument("--model", type=str, required=True, help="Type of model to train (e.g., 'GPR', 'ANN', etc.)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the input CSV data file")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target column in the data")
    parser.add_argument("--bootstrap", action="store_true", help="Use bootstrapping")
    parser.add_argument("--n_models", type=int, default=1, help="Number of models to train if bootstrapping")
    parser.add_argument("--transform_features", action="store_true", help="Apply feature transformation")
    parser.add_argument("--compound", action="store_true", help="Create compound features during transformation")
    parser.add_argument("--testsize", type=float, default=0.1, help="Test set size fraction")
    # Add more arguments as needed

    args = parser.parse_args()
    main(args)
