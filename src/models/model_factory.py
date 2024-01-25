from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Lasso
import xgboost as xgb

class ModelFactory:
    def get_model(self, model_type):
        method_name = 'create_' + model_type.lower()
        method = getattr(self, method_name, lambda: 'Invalid model type')
        return method()

    def create_krr(self):
        return KernelRidge()

    def create_ann(self):
        return MLPRegressor()

    def create_gpr(self):
        return GaussianProcessRegressor()

    def create_lasso(self):
        return Lasso()

    def create_xgb(self):
        return xgb.XGBRegressor()
