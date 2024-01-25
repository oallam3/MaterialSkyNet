import pandas as pd
import shap

class GameTheoryAnalyzer:
    def __init__(self, model, X_train, feature_cols, transform='NO'):
        self.model = model
        self.X_train = X_train
        self.feature_cols = feature_cols
        self.transform = transform

    def run(self):
        if self.transform == 'NO':
            X_train_copy = pd.DataFrame(self.X_train, columns=self.feature_cols)
            explainer = shap.Explainer(self.model.predict, X_train_copy)
        else:
            explainer = shap.Explainer(self.model.predict, self.X_train, max_evals=1000)

        shap_values = explainer(self.X_train)
        return shap_values
