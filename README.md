# MaterialSkyNet
An all in one machine learning package designed to provide key insight about feature-target relationships for any given dataset of materials. This repository houses a  framework for training, evaluating, and recalling machine learning models, with a focus on various regression techniques and ensemble methods. The framework is designed to streamline the process of model creation, hyperparameter tuning, data preprocessing, and post-hoc analysis using SHAP values.


## Overview

This code is a script for training, evaluating, and interpreting machine learning models.

- Data preprocessing: The script reads in a data file and preprocesses the data by splitting it into training and testing sets, scaling the features, and applying optional transformations to the features.
- Model training and evaluation: Support for various model types including Kernel Ridge Regression, Artificial Neural Networks, Gaussian Process Regressor, LASSO, and XGBoost. Implementation of bootstrap aggregation for ensemble modeling.
- Feature selection: The script offers an option to perform feature selection using LASSO regression, which can be applied with or without bootstrapping. After LASSO-based feature selection, a new dataset is created containing only the selected features.
- SHAP values: The script calculates SHAP values for the model to provide interpretability and understand the contributions of individual features to the model predictions.
- The script is highly customizable, with several options controlled by variables such as bootstrap, transform_features, compound, lasso_filter, search, early_stop, use_latest_model, run_shap, and others


## Dependencies

To run this script, you will need the following Python packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- shap


## Installation

Clone the repository or download the ZIP file and extract it to a directory of your choice.

git clone https://github.com/oallam3/MaterialSkyNet.git


# Usage Instructions

## Training New Models

To train a new model, use the `main.py` script with the required arguments. The script allows for various customizations such as model type selection, data file specification, target column identification, and optional features like bootstrapping, data transformation, and compound feature creation.

### Basic Command Structure:

```bash
python main.py --model MODEL_TYPE --data_file PATH_TO_DATA --target_column TARGET_COLUMN_NAME [OPTIONS]
```

Example: 
python main.py --model "ANN" --data_file "data/sample.csv" --target_column "Price" --bootstrap --n_models 5 --transform_features --compound --testsize 0.2

This command will train an Artificial Neural Network model on the 'sample.csv' dataset, predicting the 'Price' column, with 5 bootstrapped models, feature transformation, compound feature creation, and a test size of 20%.

Arguments:
--model: Type of model to train (e.g., 'GPR', 'ANN', etc.)
--data_file: Path to the input CSV data file.
--target_column: Name of the target column in the data.
--bootstrap: Use bootstrapping (optional).
--n_models: Number of models to train if bootstrapping (default is 1).
--transform_features: Apply feature transformation (optional).
--compound: Create compound features during transformation (optional).
--testsize: Test set size fraction (default is 0.1).

## Recalling Trained Models

To recall and evaluate previously trained models, use the model_recaller.py script. This script is particularly useful for models trained with bootstrapping, allowing the evaluation of each model in the ensemble.

### Basic Command Structure:

```bash
python model_recaller.py --model_name MODEL_NAME --bootstrap BOOTSTRAP --data_file PATH_TO_DATA --transform TRANSFORM --n_models NUMBER_OF_MODELS
```
Arguments:
--model_name: Name of the model to recall.
--bootstrap: Indicates if bootstrapping was used ('YES' or 'NO').
--data_file: Path to the CSV data file used for training.
--transform: Indicates if data transformation is to be applied ('YES' or 'NO').
--n_models: Number of models to recall if bootstrapping was used (relevant only if --bootstrap is 'YES').

### Example:

```bash
python model_recaller.py --model_name "ANN" --bootstrap "YES" --data_file "path/to/data.csv" --transform "NO" --n_models 10
```
This command will recall and evaluate 10 bootstrapped ANN models trained on 'data.csv', without applying any data transformation.

## Contact Information

If you have any questions or need further clarification about this project, please don't hesitate to reach out. 

**Contact Person:**
- Name: [Omar Allam]
- Email: [oallam3@gatech.edu](mailto:oallam3@gatech.edu)

We welcome your inquiries and feedback!


