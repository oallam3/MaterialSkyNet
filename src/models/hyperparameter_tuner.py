from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class HyperparameterTuner:
    def tune(self, model, X_train, y_train, search_method, hyperparams):
        if search_method == 'RS':
            return RandomizedSearchCV(model, hyperparams, n_iter=10, cv=5, n_jobs=-1).fit(X_train, y_train).best_estimator_
        else:
            return GridSearchCV(model, hyperparams, cv=5, n_jobs=-1).fit(X_train, y_train).best_estimator_
