import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from run_game import GameTheoryAnalyzer

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def standardize_features(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    return scaler.transform(X)

def tester(model_name, bootstrap, n_models, external_test_file, transform_features, run_shap, model_directory):
    df_data = pd.read_csv(external_test_file)
    X_external_test = df_data.drop(df_data.columns[-1], axis=1)
    y_external_test = df_data[df_data.columns[-1]]

    if transform_features == 'YES':
        X_external_test = standardize_features(X_external_test)

    results = []
    for i in range(n_models if bootstrap else 1):
        model_file = f"{model_directory}/{model_name}_bootstrap_{i+1}.sav" if bootstrap else f"{model_directory}/{model_name}.sav"
        model = load_model(model_file)

        mae = mean_absolute_error(y_external_test, model.predict(X_external_test))
        mse = mean_squared_error(y_external_test, model.predict(X_external_test))
        r2 = r2_score(y_external_test, model.predict(X_external_test))
        results.append((mae, mse, r2))

        if run_shap == 'YES':
            analyzer = GameTheoryAnalyzer(model, X_external_test, df_data.columns[:-1], transform=transform_features)
            shap_values = analyzer.run()
            # Save or plot SHAP values as needed

    return results


