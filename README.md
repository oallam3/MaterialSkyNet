# MaterialSkyNet
An all in one machine learning package designed to provide key insight about feature-target relationships for any given dataset of materials

This code is a script for training, evaluating, and interpreting machine learning models. The main components of the code include:

Data preprocessing: The script reads in a data file and preprocesses the data by splitting it into training and testing sets, scaling the features, and applying optional transformations to the features.
Model training and evaluation: The script supports various model types, including LASSO, ANN, and other models from the scikit-learn library. It trains the models using GridSearchCV or RandomizedSearchCV, optionally with early stopping or bootstrapping. The performance of the models is evaluated using metrics such as R2, MSE, and MAE.
Feature selection: The script offers an option to perform feature selection using LASSO regression, which can be applied with or without bootstrapping. After LASSO-based feature selection, a new dataset is created containing only the selected features.
SHAP values: The script calculates SHAP values for the model to provide interpretability and understand the contributions of individual features to the model predictions.
Plotting: The script generates various plots to visualize the model performance on training and testing sets, as well as the SHAP values for individual features.
The script is highly customizable, with several options controlled by variables such as bootstrap, transform_features, compound, lasso_filter, search, early_stop, use_latest_model, run_shap, and others

#########################################################################################################################################################

Everything under "Model Setup" in the main script (mat_skynet.py) is self explanatory. But here's the jist: This script is specifically designed to tackly the problem of statistical uncertainty pertaining to training on small/limited datasets. To circumvent this, the model will:
a) train as many bootstrapped surrogates as you'd like. This is controlled by n_models. You can turn off bootstrapping entirely by changing the "bootstrap" to "NO". I told you it was straight forward!

b) In the case of small datasets, and this is especially true for relatively simpler models such as kernel ridge regression, the model may have a difficult time mapping the complex nonlinear relations between the input features and target properties. To overcome this issue, you can augment your original features with new features which are simply the original features transformed via non-linear functions (you can play around with those in the script)...controlled via "transform_features" variable.

c) besides transforming the original features, the features can be compounded (multiplied together) to generate even more features. Note, if you implement compounding (even without transforming the original features), you will greatly expand your original feature space. This can be extremely taxing on several of the learning models.
Therefore, it is highly suggested in this case to implement the LASSO filter (controlled by, you guessed it, "lasso_filter") to downselect the features.

d) for the same reason mentioned above (to avoid statistical uncertainty that is based on the pseudorandomness of how the code selects the test/train sets), the LASSO filter can also be bootstrapped.

