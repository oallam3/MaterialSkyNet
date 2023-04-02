#....Developed by Omar Allam. 2022. v2.0
#....Please cite if using for your work.
#....You can contact me for questions or suggestions (Emails: oallam3@gatech.edu; omar-allam@hotmail.com)

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pickle
import sys, os
import time

from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.inspection import permutation_importance

from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score, mean_absolute_percentage_error
from statistics import mean

percent_error = 'NO'
if percent_error == 'YES':
    mean_absolute_error = mean_absolute_percentage_error
############################################################################
########################...Model Setup...###################################
model              = 'GPR'
use_latest_model   = 'YES' # If yes then most remaining settings irrelevant.
testsize           = 0.1
search             = 'RS' # GS (gridsearch) or RS (random search) for ANN. 
rs_samples         = 60 # relevant for RS
strat              = 'YES'
bootstrap          = 'YES'
n_models           = 100 # relevant to bootsrap aggregation
transform_features = 'NO'
compound           = 'NO'
lasso_filter       = 'NO'
lasso_filter_boot  = 'NO'
############################################################################
############################################################################

#######...Other Parameters IF you know what you are doing...################
perform_manual_GS  = 'NO'
alpha              = 'YES' #Relavent to manual_GS
alpha_range        = np.logspace(-3,1,100)
hl_range           = list(range(10,2001,10))
alpha_range_lasso  = np.logspace(-3,1,100)
model_selector     = 'TRAIN'
max_hl             = '2'
early_stop         = False
val_fraction       = 0.1 # Relevant to early stopping criteria
plot               = 'YES'
cv_n               = 5
randomSeed         = 'YES' #Automatically set to YES for bootsrap aggregation
lasso_n_models     = 2
cpu                = 6
run_shap           = 'NO'
data_file          = "data.csv"
use_external_test  = 'YES'
external_test_file = 'test_lassoed.csv'
############################################################################
if run_shap == 'YES':
    import shap
    
start = time.time()
def main(model, bootstrap, n_models, filtered_features, run_shap):

    if use_latest_model == 'NO':
        if bootstrap == 'NO':
            if randomSeed == 'NO':
                i = 0
                indices = pd.DataFrame()
                [X_train, X_test, y_train, y_test, feature_cols, indices] = data_split(data_file, i, indices)
            else:
                i = 0
                indices = pd.DataFrame()
                [X_train, X_test, y_train, y_test, feature_cols, indices] = randomSeed_data_split(data_file, i, indices)

            ######################
            #import plotly.express as px
            #train_df = pd.DataFrame(y_train)
            #test_df = pd.DataFrame(y_test)
            #fig1 = px.histogram(train_df, x='log(Kow)', height=400, title='Train Labels', nbins=30)
            #fig1.show()
            #fig2 = px.histogram(test_df, x='log(Kow)', height=400, title='Test Labels', nbins=30)
            #fig2.show()
            ######################    
            if not filtered_features == 'NULL':
                X_train = X_train[filtered_features]
                X_test  = X_test[filtered_features]
                feature_cols = filtered_features
            indices.to_csv("test_indices_"+model+".csv", index=False)
            #if not filtered_features == 'NULL':
            #    X_train = X_train[filtered_features]
            #    X_test  = X_test[filtered_features]
            #    feature_cols = filtered_features
            if lasso_filter == 'YES':
                clf, r = run_model('LASSO', X_train, X_test, y_train, y_test, feature_cols)
                print('LASSO - Screened Features')
                X_train_copy = pd.DataFrame(X_train,columns=feature_cols)
                X_test_copy = pd.DataFrame(X_test,columns=feature_cols)
                filtered_features = filter(clf, X_train_copy, 'NULL')
                print(filtered_features)
                X_train = X_train_copy[filtered_features]
                X_test = X_test_copy[filtered_features]
                feature_cols = filtered_features
                if transform_features == 'YES':
                    print(X_train)
                    filtered_features = filter(clf, X_train_copy, 'NULL')
                    #print(filtered_features)
                    #filtered_features.append(dt.columns[-1])
                    print(filtered_features)
                    #data_lasso = dt[filtered_features]
                    #data_lasso.to_csv("data_lassoed.csv", index=False)
                else:
                    dt = pd.read_csv("data.csv")
                    filtered_features = filter(clf, X_train_copy, 'NULL')
                    #print(filtered_features)
                    filtered_features.append(dt.columns[-1])
                    print(filtered_features)
                    data_lasso = dt[filtered_features]
                    data_lasso.to_csv("data_lassoed.csv", index=False)
                    filtered_features = 'NULL'
            #if lasso_filter_boot == 'YES' and transform_features == 'YES':
            #    X_train = X_train[filtered_features]
            #    X_test  = X_test[filtered_features]
            else:
                clf, r = run_model(model, X_train, X_test, y_train, y_test, feature_cols)
            pickle.dump(clf.best_estimator_, open("models/" + model+".sav", 'wb'))

                
        elif bootstrap == 'YES':
            bs_scores = []
            bs_imp = pd.DataFrame()
            indices = pd.DataFrame() ####


            for i in range(n_models):
                print("Training " + model + ": " + str(i+1))
                with HiddenPrints():
                    [X_train, X_test, y_train, y_test, feature_cols, indices] = randomSeed_data_split(data_file, i, indices)
                    if not filtered_features == 'NULL':
                        X_train = X_train[filtered_features]
                        X_test  = X_test[filtered_features]
                        feature_cols = filtered_features
                    clf, r = run_model(model, X_train, X_test, y_train, y_test, feature_cols)

                if run_shap == 'YES':
                    print('Let the Game Theory Analysis Begin!')
                    shap_values = run_game(X_train, X_test, y_train, y_test, clf, feature_cols)
                    #print(shap_values)
                    #print(type(shap_values))
                    #print(range(len(shap_values.values)))
                    if i == 0:
                        updated_shap_values_values = (shap_values.values)/n_models
                        updated_shap_values_base_values   = (shap_values.base_values)/n_models
                        updated_shap_values_data   = (shap_values.data)/n_models
                    if not (i == 0):
                        updated_shap_values_values = (shap_values.values)/n_models + updated_shap_values_values
                        updated_shap_values_base_values  = (shap_values.base_values)/n_models + updated_shap_values_base_values
                        updated_shap_values_data   = (shap_values.data)/n_models + updated_shap_values_data

                pickle.dump(clf.best_estimator_, open("models/"+model+"_bootstraps/" + model+str(i+1)+".sav", 'wb'))
                bs_scores.append((i+1, mean_absolute_error(y_train, clf.predict(X_train)), mean_absolute_error(y_test, clf.predict(X_test)), \
                                  mean_squared_error(y_train, clf.predict(X_train)), mean_squared_error(y_test, clf.predict(X_test)),\
                                  r2_score(y_train, clf.predict(X_train)),r2_score(y_test, clf.predict(X_test))))
                
                if not model == 'LASSO':
                    r.importances_mean
                    bs_imp[str(i+1)] = r.importances_mean
                else:
                    bs_imp[str(i+1)] = r
                    
            if run_shap == 'YES':
                shap_values.values = updated_shap_values_values
                shap_values.base_values = updated_shap_values_base_values
                shap_values.data = updated_shap_values_data
                shap.plots.beeswarm(shap_values, max_display=45)
                shap.plots.scatter(shap_values[:,"sim_pc"], color=shap_values)
                    #for l in range(len(shap_values.values)):
                        #if not (i == 0):
                            #for k in range(len(shap_values.data)):
                            #updated_shap_values_base_values.append(x + y for x, y in zip(shap_values.base_values[l], updated_shap_values_base_values[l]))
                            #np.append(updated_shap_values_values, (x + y for x, y in zip(shap_values.values[l], updated_shap_values_values[l])))
                            #np.append(updated_shap_values_values, shap_values.values[l] + updated_shap_values_values[l])
                            #updated_shap_values_values, shap_values.values[l] + updated_shap_values_values[l])
                            #shap_values.values[l][k]
                            #print('added')
                    #print('updated array')
                    #print(updated_shap_values_values)
                    #print(updated_shap_values_base_values)
                    #print(updated_shap_values_data)
                    
            
            bs_scores=pd.DataFrame(bs_scores, columns=('Model Number', 'Train MAE', 'Test MAE', 'Train MSE', 'Test MSE','Train R2', 'Test R2'))
            bs_scores[' ']=' '
            #bs_scores[['Avg. Train MSE', 'Avg. Test MSE','Avg. Train R2', 'Avg. Test R2']] = \
            #                [bs_scores['Train MSE'].mean(), bs_scores['Test MSE'].mean(), bs_scores['Train R2'].mean(), bs_scores['Test R2'].mean()]
            bs_imp['Average Importance'] = bs_imp.mean(axis=1)
            bs_imp.insert(0, 'Feature', feature_cols)
            bs_imp = bs_imp.sort_values('Average Importance')
            bs_imp['Standard Deviation'] = bs_imp.std(axis=1)
            if transform == 'YES':
                if compound == 'YES':
                    bs_scores.to_csv("scores_"+model+"_"+str(n_models)+"_random_runs_trans_compound.csv")
                    bs_imp.to_csv("feature_importance_"+model+"_"+str(n_models)+"_random_runs_trans_compound.csv")
                elif compound == 'NO':
                    bs_scores.to_csv("scores_"+model+"_"+str(n_models)+"_random_runs_trans.csv")
                    bs_imp.to_csv("feature_importance_"+model+"_"+str(n_models)+"_random_runs_trans.csv")
            else:
                bs_scores.to_csv("scores_"+model+"_"+str(n_models)+"_random_runs.csv")
                bs_imp.to_csv("feature_importance_"+model+"_"+str(n_models)+"_random_runs.csv")
            print(' ')
            print(indices)
            indices.to_csv("test_indices_"+model+"_"+str(n_models)+"_random_runs.csv", index=False)
            print(' ')
            print(bs_scores)
            print(' ')
            print("Avg. Test R2: "+str(bs_scores['Test R2'].mean()))
            print("Avg. Test MSE: "+str(bs_scores['Test MSE'].mean()))
            print("Avg. Test MAE: "+str(bs_scores['Test MAE'].mean()))
            print(' ')
            print("Avg. Train R2: "+str(bs_scores['Train R2'].mean()))
            print("Avg. Train MSE: "+str(bs_scores['Train MSE'].mean()))
            print("Avg. Train MSE: "+str(bs_scores['Train MAE'].mean()))
            print(' ')
            print(bs_imp)
            if lasso_filter == 'YES':
                if transform_features == 'YES':
                    #dt = pd.read_csv("data.csv")
                    filtered_features = filter('NULL', X_train, bs_imp)
                    print(filtered_features)
                    #filtered_features.append(dt.columns[-1])
                    #data_lasso = X_train[filtered_features]
                    with open('filtered_trans_features.txt', 'w') as filehandle:
                        for listitem in filtered_features:
                            filehandle.write(f'{listitem}\n')
                else:
                    dt = pd.read_csv("data.csv")
                    filtered_features = filter('NULL', dt, bs_imp)
                    #print(filtered_features)
                    filtered_features.append(dt.columns[-1])
                    print(filtered_features)
                    data_lasso = dt[filtered_features]
                    data_lasso.to_csv("data_lassoed.csv", index=False)
                    filtered_features = 'NULL'

    elif use_latest_model == 'YES':
        if use_external_test == 'NO':
            X_train_recalled, X_test_recalled, y_train_recalled, y_test_recalled = recaller(model, bootstrap, lasso_filter_boot, n_models, data_file, filtered_features)
        else:
            X_external_test, y_external_test = tester(model, bootstrap, lasso_filter_boot, n_models, external_test_file, filtered_features)
        
    end = time.time()
    print("Elaspsed Time: " + str(round(end - start, 2)) + " s")
    print("Elaspsed Time: " + str(round((end - start)/3600, 2)) + " hr")    

    if not lasso_filter == 'YES':   
        if plot == 'YES':
            if use_latest_model == 'YES':
                plotter(X_train_recalled, X_test_recalled, y_train_recalled, y_test_recalled)
            else:
                plotter(X_train, X_test, y_train, y_test)

    return filtered_features

def run_model(model, X_train, X_test, y_train, y_test, feature_cols):

    if model == 'KRR':
        ml = KernelRidge()
        tuned_parameters = [{'kernel':["linear","rbf"],'alpha': alpha_range}]
    elif model == 'ANN':
        ml = MLPRegressor()
        hl = []
        for i in range(1, 501):
            hl.append((i,))
        for i in range(1, 501):
            for j in range(1, 501):
                hl.append((i,j))
        tuned_parameters = {"hidden_layer_sizes": hl,"max_iter":[10000],\
                            "activation": ['logistic', 'tanh', 'relu'], "solver": [ "adam"], "alpha": alpha_range, \
                            "learning_rate": ["constant", "adaptive"], "random_state": [42], "early_stopping": [early_stop],\
                            "validation_fraction": [val_fraction], "shuffle": [False]}
    elif model == 'LASSO':
        ml = linear_model.Lasso()
        tuned_parameters = [{'alpha': alpha_range_lasso,"max_iter":[10000000]}]
    elif model == 'GPR':
        #kernel = RBF(10, [1e-2, 1e2]) + WhiteKernel(noise_level=[1e-2, 1e2])
        #kernel = RBF(10) + WhiteKernel()
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-3, 1e3)) + WhiteKernel(1.0, (1e-5, 1e3))
        ml = GaussianProcessRegressor()
        tuned_parameters = {'kernel':[kernel], 'n_restarts_optimizer':[10], 'alpha': alpha_range}
    elif model == 'XGB':
        ml = xgb.XGBRegressor()
        tuned_parameters = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200], 'alpha': alpha_range}
                            #'predictor': ['gpu_predictor'], 'tree_method': ['gpu_hist']}
        

    if perform_manual_GS == 'YES':
        [tuned_parameters, clf] = manual_GS(model,alpha,alpha_range, X_train, X_test, y_train, y_test)
    elif perform_manual_GS == 'NO':
        print("Starting " + model + " Hyperparameter Optimization")
        print(" ")
        if search == "RS":
            clf = RandomizedSearchCV(ml, tuned_parameters, cv=5, n_iter=rs_samples, n_jobs=cpu)
        elif search == "GS":
            if model == 'ANN':
                hl = []
                for i in range(25, 501,25):
                    hl.append((i,))
                for i in range(25, 501,25):
                    for j in range(1, 501,25):
                        hl.append((i,j))
            clf = GridSearchCV(ml, tuned_parameters, cv=cv_n, n_jobs=cpu)
            
        #print(clf.best_params_)
        clf.fit(X_train, y_train)
        print("Optimal model so far: ")
        print(clf.best_estimator_)

    #if model == 'LASSO':
        

    
   

    #print(clf.best_params_)
    print(" ")
    y_pred = clf.predict(X_test)
    print("Test R2: " + str(r2_score(y_test, y_pred)))
    print("Test MSE: " + str(mean_squared_error(y_test, y_pred)))
    print(" ")
    y_pred_train = clf.predict(X_train)
    print("Train R2: " + str(r2_score(y_train, y_pred_train)))
    print("Train MSE: " + str(mean_squared_error(y_train, y_pred_train)))

    if not model == 'LASSO':
        ####Permutation Importance#######
        #model = clf.fit(X_train, y_train)
        r = permutation_importance(clf.best_estimator_, X_train, y_train,
                                   n_repeats=30,
                                   random_state=0)
        
        #X_train_copy = pd.DataFrame(X_train,columns=X_train.columns[:])
        X_train_copy = pd.DataFrame(X_train,columns=feature_cols)
        
        #y_train = pd.DataFrame(y_train,columns=[feature_out])

        imp_arr = []
        feature_arr = []

        for i in r.importances_mean.argsort()[::-1]:
    ##        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    ##            print(f"{X_train_copy.columns[i]}"
    ##                  f"{r.importances_mean[i]:.3f}"
    ##                  f" +/- {r.importances_std[i]:.3f}")
                imp = r.importances_mean[i]
                imp_arr.append(imp)
                feature_arr.append(X_train_copy.columns[i])
        
        feature_importance = imp_arr

        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        feature_arr = [feature_arr[i] for i in sorted_idx]

        feature_cols_sorted = feature_arr
        feature_importance.sort()

        f_imp = pd.DataFrame()
        f_imp["feature"] = feature_cols_sorted
        print(" ")
        f_imp["importance"] = feature_importance
        f_imp["%"] = 100.0 * (feature_importance / max(feature_importance))
        print("----------------------------------")
        print("----------------------------------")
        print("----------------------------------")
        print(f_imp)
        print("----------------------------------")
        print("----------------------------------")
        print("----------------------------------")

        #explainer = shap.Explainer(clf.predict, X_train_copy)
        #shap_values = explainer(X_train_copy)
        #shap.plots.beeswarm(shap_values, max_display=50)
        #shap.plots.waterfall(shap_values[0])
        #shap.plots.waterfall(shap_values[3])
        ##shap.plots.scatter(shap_values[:,"sim_pc"], color=shap_values)


        if bootstrap == 'NO':
            f_imp.to_csv("feature_importance_"+model+".csv")
    elif model == 'LASSO':
        r = clf.best_estimator_.coef_
        if bootstrap == 'NO':
            d = {'Feature': feature_cols, 'LASSO Coefficient': r}
            f_coeff = pd.DataFrame(data=d)
            f_coeff = f_coeff.sort_values('LASSO Coefficient')
            f_coeff.to_csv("lasso_coefficients_"+model+".csv")
        
        

    #np.savetxt("res/krr_compound_res.csv", y_pred, delimiter=",")

    #res_test = pd.DataFrame(index=y_train.index )
    #res_test["y"] = y_train
    #res_test["pred(y)"] = clf.predict(X_train)
    #res_test.to_csv("res/krr_compound_res_train.csv")

    

    return(clf, r)
##############################################################################################################################    
##############################################################################################################################
##############################################################################################################################

def run_game(X_train, X_test, y_train, y_test, clf, feature_cols):

    if transform == 'NO':
        X_train_copy = pd.DataFrame(X_train,columns=feature_cols)
        X_test_copy = pd.DataFrame(X_test,columns=feature_cols)
        explainer = shap.Explainer(clf.predict, X_train_copy)
        shap_values = explainer(X_train_copy)
    else:
        print(X_train)
        explainer = shap.Explainer(clf.predict, X_train, max_evals=1000)
        shap_values = explainer(X_train)
    #print(shap_values.values)
    #print(type(shap_values.values))
    #print(shap_values.values[1])
    #print(type(shap_values.values[1]))
    #shap.plots.beeswarm(shap_values, max_display=10)
    #shap.plots.waterfall(shap_values[0])
    #shap.plots.waterfall(shap_values[3])
    ##shap.plots.scatter(shap_values[:,"sim_pc"], color=shap_values)

    return shap_values


def manual_GS(model,alpha,alpha_range, X_train, X_test, y_train, y_test):
    
    print("Initiating manual gridsearch of alpha. This may take a while...")
    print(' ')
    
    perf_alpha = []

    if alpha == 'YES':
        incumbent_mse = 1000
        for x in alpha_range:
            if model == 'KRR':
                ml = KernelRidge()
                tuned_parameters = [{'kernel':["rbf"],'alpha': [x]}]
            elif model == 'ANN':
                ml = MLPRegressor()
                tuned_parameters = {"hidden_layer_sizes":[(100,)],"max_iter":[10000], "activation": ["logistic"], "solver": [ "adam"], "alpha": [x]}
            #ml = KernelRidge()
            #print("Starting CV")
            clf = GridSearchCV(ml, tuned_parameters, cv=5, n_jobs=cpu)
            clf.fit(X_train, y_train)
            #print(clf.best_params_)
            
            #print("Test Set MSE + R2:")
            y_pred = clf.predict(X_test)
            #print(mean_squared_error(y_test, y_pred))
            #print(r2_score(y_test, y_pred))
            #print("Training Set MSE + R2:")
            y_pred_train = clf.predict(X_train)
            #print(mean_squared_error(y_train, y_pred_train))
            #print(r2_score(y_train, y_pred_train))
            perf_alpha.append([x, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_train, y_pred_train), r2_score(y_train, y_pred_train)])
            if model_selector == 'TRAIN':
                if mean_squared_error(y_train, y_pred_train) < incumbent_mse:
                    best_so_far_clf_alpha_krr = pickle.dumps(clf)
                    incumbent_mse = mean_squared_error(y_train, y_pred_train)
            elif model_selector == 'TEST':
                if mean_squared_error(y_test, y_pred) < incumbent_mse:
                    best_so_far_clf_alpha_krr = pickle.dumps(clf)
                    incumbent_mse = mean_squared_error(y_test, y_pred)
        GS = pd.DataFrame(perf_alpha, columns = ['alpha','MSE - Test','R2 - Test','MSE - Train','R2 - Train'])
        if model == 'ANN':
            GS.to_csv("res/Manual_ANN_alpha_GridSearch.csv")
        elif model == 'KRR':
            GS.to_csv("res/Manual_KRR_alpha_GridSearch.csv")
        opt_alpha = GS['alpha'].iloc[GS['MSE - Train'].idxmin()]
        opt_alpha_test = GS['alpha'].iloc[GS['MSE - Test'].idxmin()]
        print("Optimal " + model + " Alpha (Based on train set) = " + str(opt_alpha))
        print(' ')
        print("Optimal " + model + " Alpha (Based on test set) = " + str(opt_alpha_test))
        print(' ')
        if model_selector == 'TEST':
            opt_alpha = opt_alpha_test


    perf_1hl = []
    perf_2hl = []
    incumbent_mse = 1000
    if model == 'ANN':
        if hlgs == 'YES':
            print("Initiating manual hidden layer gridsearch. This may take a while...")
            print(' ')
            for k in hl_range:
                    ml = MLPRegressor()
                    tuned_parameters = {"hidden_layer_sizes":[(k,)],"max_iter":[10000], "activation": \
                                        ["logistic"], "solver": [ "adam"], "alpha": [opt_alpha]}
                
                    #ml = KernelRidge()
                    #print("Starting CV")
                    clf = GridSearchCV(ml, tuned_parameters, cv=5, n_jobs=cpu)
                    clf.fit(X_train, y_train)
                    #print(clf.best_params_)
                    
                    #print("Test Set MSE + R2:")
                    y_pred = clf.predict(X_test)
                    #print(mean_squared_error(y_test, y_pred))
                    #print(r2_score(y_test, y_pred))
                    #print("Training Set MSE + R2:")
                    y_pred_train = clf.predict(X_train)
                    #print(mean_squared_error(y_train, y_pred_train))
                    #print(r2_score(y_train, y_pred_train))
                    #perf.append([x, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error\
                    #(y_train, y_pred_train), r2_score(y_train, y_pred_train)])
                    perf_1hl.append([k, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error\
                                  (y_train, y_pred_train), r2_score(y_train, y_pred_train)])
                    if model_selector == 'TRAIN':
                        if mean_squared_error(y_train, y_pred_train) < incumbent_mse:
                                best_so_far_clf_1hl = pickle.dumps(clf)
                                incumbent_mse = mean_squared_error(y_train, y_pred_train)
                    elif model_selector == 'TEST':
                        if mean_squared_error(y_test, y_pred) < incumbent_mse:
                                best_so_far_clf_1hl = pickle.dumps(clf)
                                incumbent_mse = mean_squared_error(y_test, y_pred)
            HLGS1 = pd.DataFrame(perf_1hl, columns = ['HL','MSE - Test','R2 - Test','MSE - Train','R2 - Train'])
            HLGS1.to_csv("res/Manual_1_HiddenLayer_GridSearch.csv")
            #print(HLGS1)
            mse_1HL = HLGS1['MSE - Train'].idxmin()
            mse_1HL_test = HLGS1['MSE - Test'].idxmin()           
            #print(str(mse_1HL))
            opt_hl = HLGS1['HL'].iloc[HLGS1['MSE - Train'].idxmin()]
            opt_hl_test = HLGS1['HL'].iloc[HLGS1['MSE - Test'].idxmin()]
            print("Optimal [HL,] (based on train set) = " + str(opt_hl))
            print("Optimal [HL,] (based on test  set) = " + str(opt_hl_test))
            print(' ')
            
            if max_hl == '2':
                for i in hl_range:
                    for j in hl_range:
                        ml = MLPRegressor()
                        tuned_parameters = {"hidden_layer_sizes":[(i,j)],"max_iter":[10000], "activation": ["logistic"],\
                                            "solver": [ "adam"], "alpha": [opt_alpha]}
                    
                        #ml = KernelRidge()
                        #print("Starting CV")
                        clf = GridSearchCV(ml, tuned_parameters, cv=5, n_jobs=cpu)
                        clf.fit(X_train, y_train)
                        #print(clf.best_params_)
                        
                        #print("Test Set MSE + R2:")
                        y_pred = clf.predict(X_test)
                        #print(mean_squared_error(y_test, y_pred))
                        #print(r2_score(y_test, y_pred))
                        #print("Training Set MSE + R2:")
                        y_pred_train = clf.predict(X_train)
                        #print(mean_squared_error(y_train, y_pred_train))
                        #print(r2_score(y_train, y_pred_train))
                        #perf.append([x, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error\
                        #(y_train, y_pred_train), r2_score(y_train, y_pred_train)])
                        perf_2hl.append([i, j, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error\
                                      (y_train, y_pred_train), r2_score(y_train, y_pred_train)])
                        if model_selector == 'TRAIN':
                            if mean_squared_error(y_train, y_pred_train) < incumbent_mse:
                                best_so_far_clf_2hl = pickle.dumps(clf)
                                incumbent_mse = mean_squared_error(y_train, y_pred_train)
                        elif model_selector == 'TEST':
                            if mean_squared_error(y_test, y_pred) < incumbent_mse:
                                best_so_far_clf_2hl = pickle.dumps(clf)
                                incumbent_mse = mean_squared_error(y_test, y_pred)
                        
                HLGS2 = pd.DataFrame(perf_2hl, columns = ['HL1','HL2','MSE - Test','R2 - Test','MSE - Train','R2 - Train'])
                HLGS2.to_csv("res/Manual_2_HiddenLayers_GridSearch.csv")
                #print(HLGS2)
                mse_2HL = HLGS2['MSE - Train'].idxmin()
                mse_2HL_test = HLGS2['MSE - Test'].idxmin()
                opt_hl1 = HLGS2['HL1'].iloc[HLGS2['MSE - Train'].idxmin()]
                opt_hl2 = HLGS2['HL2'].iloc[HLGS2['MSE - Train'].idxmin()]

                opt_hl1_test = HLGS2['HL1'].iloc[HLGS2['MSE - Test'].idxmin()]
                opt_hl2_test = HLGS2['HL2'].iloc[HLGS2['MSE - Test'].idxmin()]
                print("Optimal [HL1, HL2] (based on train set) = " + "[" + str(opt_hl1) + ", " + str(opt_hl2) + "]")
                print("Optimal [HL1, HL2] (based on test  set) = " + "[" + str(opt_hl1) + ", " + str(opt_hl2) + "]")
                print(' ')



    if model == 'ANN':
        if model_selector == 'TEST':
            if mse_1HL_test <= mse_2HL_test:
                print("(MSE of 1 HL = " + str(mse_1HL_test) +\
                      ")  <=  (MSE of 2 HL = " + str(mse_2HL_test) + "), 1 hidden layer is optimal")
                tuned_parameters = {"hidden_layer_sizes":[(opt_hl_test,)],\
                                    "max_iter":[10000], "activation": ["logistic"], "solver": [ "adam"], "alpha": [opt_alpha]}
                clf = pickle.loads(best_so_far_clf_1hl)
            elif mse_1HL_test > mse_2HL_test:
                print("(MSE of 1 HL = " + str(mse_1HL_test) +\
                      ")  >  (MSE of 2 HL = " + str(mse_2HL_test) + "), 2 hidden layers are optimal") 
                tuned_parameters = {"hidden_layer_sizes":[(opt_hl1_test, opt_hl2_test)]\
                                    ,"max_iter":[10000], "activation": ["logistic"], "solver": [ "adam"], "alpha": [opt_alpha]}
                clf = pickle.loads(best_so_far_clf_2hl)
        elif model_selector == 'TRAIN':
            if mse_1HL <= mse_2HL:
                print("(MSE of 1 HL = " + str(mse_1HL) +\
                      ")  <=  (MSE of 2 HL = " + str(mse_2HL) + "), 1 hidden layer is optimal") 
                tuned_parameters = {"hidden_layer_sizes":[(opt_hl,)],\
                                    "max_iter":[10000], "activation": ["logistic"], "solver": [ "adam"], "alpha": [opt_alpha]}
                clf = pickle.loads(best_so_far_clf_1hl)
            elif HLGS1['MSE - Train'].idxmin() > HLGS2['MSE - Train'].idxmin():
                print("(MSE of 1 HL = " + str(mse_1HL) +\
                      ")  >  (MSE of 2 HL = " + str(mse_2HL) + "), 2 hidden layers are optimal") 
                tuned_parameters = {"hidden_layer_sizes":[(opt_hl1, opt_hl2)],\
                                    "max_iter":[10000], "activation": ["logistic"], "solver": [ "adam"], "alpha": [opt_alpha]}
                clf = pickle.loads(best_so_far_clf_1hl)
        print(' ')
    elif model == 'KRR':
        tuned_parameters = [{'kernel':["rbf"],'alpha': [opt_alpha]}]
        clf = pickle.loads(best_so_far_clf_alpha_krr)

    print("Manual gridsearch complete")
    print(" ")
    
    return tuned_parameters, clf

def transform(model,X_train, X_test, y_train, y_test, feature_cols):
    ##############
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    X_test_  = scaler.transform(X_test)


    #You can try MinMaxScaler. In the past, I believe we used MaxAbsScaler and it worked well.
    #scaler = MinMaxScaler((-1, 1))  
    scaler = MaxAbsScaler()  
    scaler.fit(X_train_)
    X_train_ = scaler.transform(X_train_)
    X_test_  = scaler.transform(X_test_)

    X_train_ = pd.DataFrame(X_train_, columns = feature_cols)
    X_test_  = pd.DataFrame(X_test_, columns = feature_cols)


    #X_train_.to_csv("X_train_scaled.csv")
    #X_test_.to_csv("X_test_scaled.csv")

    ################


    X_train_trans = X_train_
    X_test_trans = X_test_


    for l, feature in enumerate(feature_cols):
        X_train_trans[feature + '^2'] = X_train_.loc[:, feature]**2
        X_test_trans[feature + '^2'] = X_test_.loc[:, feature]**2

    for l, feature in enumerate(feature_cols):
        X_train_trans['(' + feature + '+1)^1/2'] = (X_train_.loc[:, feature]+1) ** (1/2)
        X_test_trans['(' + feature + '+1)^1/2'] = (X_test_.loc[:, feature]+1) ** (1/2)


    for l, feature in enumerate(feature_cols):
        X_train_trans['log(' + feature + '+2)'] = (np.log10(X_train_.loc[:, feature] + 2))
        X_test_trans['log(' + feature + '+2)'] = (np.log10(X_test_.loc[:, feature] + 2))

    for l, feature in enumerate(feature_cols):
        X_train_trans['e^(' + feature + ')'] = (np.exp(X_train_.loc[:, feature]))
        X_test_trans['e^(' + feature + ')'] = (np.exp(X_test_.loc[:, feature]))

    #Some of the scaled features in testing set will exceed the scaler limits since it was set on training set.
    #So the functions will yield a NA if they are -ve or 0
    for feature in X_test_trans.columns:
        X_test_trans[feature].fillna(X_test_trans[feature].astype('float32').mean(),inplace=True)

    #X_train_trans.to_csv("res/X_train_trans.csv")
    #X_test_trans.to_csv("res/X_test_trans.csv")

    #print(X_train_trans)

    if compound == "YES":
        for l, feature in enumerate(X_train_trans.columns):
            for i, f in enumerate(X_train_trans.columns):
                if i >= l:
                    pass
                else:
                    X_train_trans[feature + '*' + f] = X_train_trans.loc[:, feature]*X_train_trans.loc[:, f]
                    X_test_trans[feature + '*' + f] = X_test_trans.loc[:, feature]*X_test_trans.loc[:, f]     
##    print(X_train_trans)
##    print(X_test_trans)
    if bootstrap == 'NO':
        X_train_trans.to_csv("res/X_train_trans_comp.csv")
        X_test_trans.to_csv("res/X_test_trans_comp.csv")



    #print(X_test)

    feature_cols_trans = X_train_trans.columns[:]
    print(X_test_trans)
    
    return X_train_trans, X_test_trans, feature_cols_trans

def stratifier(data_file):
    bin_count = 5
    bin_numbers = pd.qcut(x=pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[-1]], q=bin_count, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[:-1]], \
                                                        pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[-1]], \
                                                        test_size=testsize, random_state=42, stratify=bin_numbers)
    return X_train, X_test, y_train, y_test

def randomSeed_stratifier(data_file):
    bin_count = 5
    bin_numbers = pd.qcut(x=pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[-1]], q=bin_count, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[:-1]], \
                                                        pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[-1]], \
                                                        test_size=testsize, stratify=bin_numbers)
    return X_train, X_test, y_train, y_test

def data_split(data_file, i, indices):
    if strat == 'YES':
        [X_train, X_test, y_train, y_test] = stratifier(data_file)
    elif strat == 'NO':
        X_train, X_test, y_train, y_test = train_test_split(pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[:-1]], \
                                                        pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[-1]], \
                                                        test_size=testsize, random_state=42)
    indices[i+1] = X_test.index.values.tolist()  ####
    
    feature_cols = X_train.columns[:]
    if transform_features == 'YES':
        X_train_trans, X_test_trans, feature_cols_trans = transform(model,X_train, X_test, y_train, y_test,feature_cols)
        X_train = X_train_trans
        X_test  = X_test_trans
        feature_cols = feature_cols_trans
    else:
        scaler = StandardScaler()  
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)
    print(' ')
    return X_train, X_test, y_train, y_test, feature_cols, indices

def randomSeed_data_split(data_file, i, indices):
    if strat == 'YES':
        [X_train, X_test, y_train, y_test] = randomSeed_stratifier(data_file)
    elif strat == 'NO':
        X_train, X_test, y_train, y_test = train_test_split(pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[:-1]], \
                                                        pd.read_csv(data_file).loc[:, pd.read_csv(data_file).columns[-1]], \
                                                        test_size=testsize)
    indices[i+1] = X_test.index.values.tolist()  ####
    
    feature_cols = X_train.columns[:]
    if transform_features == 'YES':
        X_train_trans, X_test_trans, feature_cols_trans = transform(model,X_train, X_test, y_train, y_test,feature_cols)
        X_train = X_train_trans
        X_test  = X_test_trans
        feature_cols = feature_cols_trans
    else:
        scaler = StandardScaler()  
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)
    print(' ')
    return X_train, X_test, y_train, y_test, feature_cols, indices

################
    
def recaller(model, bootstrap, lasso_filter_boot, n_models, data_file, filtered_features):
    df_data = pd.read_csv(data_file)

    if bootstrap == 'YES':
        test_ind = pd.read_csv("test_indices_"+model+"_"+str(n_models)+"_random_runs.csv")
    else:
        test_ind = pd.read_csv("test_indices_"+model+".csv")
        n_models = 1
        
    mse_recalled_train = []
    mse_recalled_test = []
    mae_recalled_train = []
    mae_recalled_test = []
    r2_recalled_train = []
    r2_recalled_test = []
    for i in range(n_models):
        print('Recalling Model No. ' + str(i+1)+ "/"+str(n_models))
        X_train_recalled = df_data.drop(test_ind[str(i+1)].values.tolist()).drop(df_data.columns[-1], axis=1)
        y_train_recalled = pd.DataFrame(df_data.drop(test_ind[str(i+1)].values.tolist())[df_data.columns[-1]], columns = [df_data.columns[-1]])

        X_test_recalled = df_data.iloc[test_ind[str(i+1)].values.tolist()].drop(df_data.columns[-1], axis=1)
        y_test_recalled = pd.DataFrame(df_data.iloc[test_ind[str(i+1)].values.tolist()][df_data.columns[-1]], columns = [df_data.columns[-1]])

        feature_cols = X_train_recalled.columns[:]
        if transform_features == 'YES':
            X_train_trans, X_test_trans, feature_cols_trans\
                           = transform(model,X_train_recalled, X_test_recalled, y_train_recalled, y_test_recalled,feature_cols)
            X_train_recalled = X_train_trans
            X_test_recalled  = X_test_trans
            feature_cols = feature_cols_trans
            if not filtered_features == 'NULL':
                X_train_recalled = X_train_recalled[filtered_features]
                X_test_recalled  = X_test_recalled[filtered_features]
                feature_cols = filtered_features
        else:
            scaler = StandardScaler()  
            scaler.fit(X_train_recalled)
            X_train_recalled = scaler.transform(X_train_recalled)
            X_test_recalled  = scaler.transform(X_test_recalled)



        if bootstrap == 'YES':
            clf = pickle.load(open("models/"+model+"_bootstraps/" + model+str(i+1)+".sav", 'rb'))
        else:
            clf = pickle.load(open("models/" + model+".sav", 'rb'))
            
        mae_recalled_train.append(mean_absolute_error(y_train_recalled, clf.predict(X_train_recalled)))
        mae_recalled_test.append(mean_absolute_error(y_test_recalled, clf.predict(X_test_recalled)))
        mse_recalled_train.append(mean_squared_error(y_train_recalled, clf.predict(X_train_recalled)))
        mse_recalled_test.append(mean_squared_error(y_test_recalled, clf.predict(X_test_recalled)))
        r2_recalled_train.append(r2_score(y_train_recalled, clf.predict(X_train_recalled)))
        r2_recalled_test.append(r2_score(y_test_recalled, clf.predict(X_test_recalled)))

        ##########################
        if run_shap == 'YES':
            print('Let the Game Theory Analysis Begin! — ' + str(i+1))
            shap_values = run_game(X_train_recalled, X_test_recalled, y_train_recalled, y_test_recalled, clf, feature_cols)
            #print(shap_values)
            #print(type(shap_values))
            #print(range(len(shap_values.values)))
            if i == 0:
                updated_shap_values_values = (shap_values.values)/n_models
                updated_shap_values_base_values   = (shap_values.base_values)/n_models
                updated_shap_values_data   = (shap_values.data)/n_models
            if not (i == 0):
                updated_shap_values_values = (shap_values.values)/n_models + updated_shap_values_values
                updated_shap_values_base_values  = (shap_values.base_values)/n_models + updated_shap_values_base_values
                updated_shap_values_data   = (shap_values.data)/n_models + updated_shap_values_data
    if run_shap == 'YES':
        shap_values.values = updated_shap_values_values
        shap_values.base_values = updated_shap_values_base_values
        shap_values.data = updated_shap_values_data
        shap.plots.beeswarm(shap_values, max_display=45)

        if transform == 'NO' and compound == 'NO':
            shap.plots.scatter(shap_values[:,"sim_pc"], color=shap_values)
            shap.plots.scatter(shap_values[:,"HOMO"], color=shap_values)
            shap.plots.scatter(shap_values[:,"HOMO"], color=shap_values[:,"sim_pc"])
            shap.plots.scatter(shap_values[:,"sfe_water"], color=shap_values[:,"sim_pc"])
            shap.plots.scatter(shap_values[:,"sfe_water"], color=shap_values)
            shap.plots.scatter(shap_values[:,"sfe_octa"], color=shap_values)
            shap.plots.scatter(shap_values[:,"neg_partial"], color=shap_values)
            shap.plots.scatter(shap_values[:,"pos_partial"], color=shap_values)
            shap.plots.scatter(shap_values[:,"mag_quad YY"], color=shap_values)
            shap.plots.scatter(shap_values[:,"LUMO"])
            shap.plots.scatter(shap_values[:,"num_benzene"])
            shap.plots.scatter(shap_values[:,"LUMO"], color=shap_values)
            shap.plots.scatter(shap_values[:,"num_benzene"], color=shap_values)
        elif transform == 'YES' and compound == 'YES':
            shap.plots.scatter(shap_values[:,"e^(sfe_octa)*e^(sfe_water)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(sim_pc)*e^(sfe_water)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(sfe_octa)*e^(sfe_water)"], color=shap_values[:,"e^(sim_pc)*e^(sfe_water)"])
            shap.plots.scatter(shap_values[:,"e^(num_o)*(HOMO+1)^1/2"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(neg_partial)*e^(sfe_water)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(mag_quad ZZ)*e^(mag_quad YY)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(mag_quad YY)*e^(mag_quad XX)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"mag_quad YZ*mag_quad XX"])
            shap.plots.scatter(shap_values[:,"e^(neg_partial)*(mag_quad YY+1)^1/2"])
            shap.plots.scatter(shap_values[:,"e^(mag_dipole)*ent^2"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(num_o)*mag_quad XZ^2"], color=shap_values)



        with open('shap_values.values' + model+'.txt', 'w') as filehandle:
            for listitem in shap_values.values:
                filehandle.write(f'{listitem}\n')

        with open('shap_values.base_values' + model+'.txt', 'w') as filehandle:
            for listitem in shap_values.base_values:
                filehandle.write(f'{listitem}\n')

        with open('shap_values.data' + model+'.txt', 'w') as filehandle:
            for listitem in shap_values.data:
                filehandle.write(f'{listitem}\n')
################################
    if bootstrap == 'YES':
        print("********Ensemble Recaller Activated********")
        print("Avg. Test R2: "+str(mean(r2_recalled_test)))
        print("Avg. Test MSE: "+str(mean(mse_recalled_test)))
        print("Avg. Test MAE: "+str(mean(mae_recalled_test)))
        print(' ')
        print("Avg. Train R2: "+str(mean(r2_recalled_train)))
        print("Avg. Train MSE: "+str(mean(mse_recalled_train)))
        print("Avg. Train MAE: "+str(mean(mae_recalled_train)))
    else:
        print("********Recaller Activated********")
        print("Test R2: "+str(mean(r2_recalled_test)))
        print("Test MSE: "+str(mean(mse_recalled_test)))
        print(' ')
        print("Train R2: "+str(mean(r2_recalled_train)))
        print("Train MSE: "+str(mean(mse_recalled_train)))

    return X_train_recalled, X_test_recalled, y_train_recalled, y_test_recalled

#######################################
#########################################
###################################

    
def tester(model, bootstrap, lasso_filter_boot, n_models, external_test_file, filtered_features):
    df_data = pd.read_csv(external_test_file)
    X_external_test = df_data.drop(df_data.columns[-1], axis=1)
    y_external_test = pd.DataFrame(df_data[df_data.columns[-1]], columns = [df_data.columns[-1]])

    feature_cols = X_external_test.columns[:]
        
    mse_external_test = []
    mae_external_test = []
    r2_external_test = []
    
    for i in range(n_models):
        print('Testing No. ' + str(i+1)+ "/"+str(n_models))

        if transform_features == 'YES':
            X_train_trans, X_test_trans, feature_cols_trans\
                           = transform(model,X_train_recalled, X_external_test, y_train_recalled, y_external_test,feature_cols)
            X_train_recalled = X_train_trans
            X_external_test  = X_test_trans
            feature_cols = feature_cols_trans
            if not filtered_features == 'NULL':
                X_train_recalled = X_train_recalled[filtered_features]
                X_external_test  = X_external_test[filtered_features]
                feature_cols = filtered_features
        else:
            scaler = StandardScaler()  
            scaler.fit(X_external_test)
            X_external_test  = scaler.transform(X_external_test)



        if bootstrap == 'YES':
            clf = pickle.load(open("models/"+model+"_bootstraps/" + model+str(i+1)+".sav", 'rb'))
        else:
            clf = pickle.load(open("models/" + model+".sav", 'rb'))
            
        mae_external_test.append(mean_absolute_error(y_external_test, clf.predict(X_external_test)))
        mse_external_test.append(mean_squared_error(y_external_test, clf.predict(X_external_test)))
        r2_external_test.append(r2_score(y_external_test, clf.predict(X_external_test)))

##########################
        if run_shap == 'YES':
            print('Let the Game Theory Analysis Begin! — ' + str(i+1))
            shap_values = run_game(X_train_recalled, X_test_recalled, y_train_recalled, y_test_recalled, clf, feature_cols)
            #print(shap_values)
            #print(type(shap_values))
            #print(range(len(shap_values.values)))
            if i == 0:
                updated_shap_values_values = (shap_values.values)/n_models
                updated_shap_values_base_values   = (shap_values.base_values)/n_models
                updated_shap_values_data   = (shap_values.data)/n_models
            if not (i == 0):
                updated_shap_values_values = (shap_values.values)/n_models + updated_shap_values_values
                updated_shap_values_base_values  = (shap_values.base_values)/n_models + updated_shap_values_base_values
                updated_shap_values_data   = (shap_values.data)/n_models + updated_shap_values_data
    if run_shap == 'YES':
        shap_values.values = updated_shap_values_values
        shap_values.base_values = updated_shap_values_base_values
        shap_values.data = updated_shap_values_data
        shap.plots.beeswarm(shap_values, max_display=45)

        if transform == 'NO' and compound == 'NO':
            shap.plots.scatter(shap_values[:,"sim_pc"], color=shap_values)
            shap.plots.scatter(shap_values[:,"HOMO"], color=shap_values)
            shap.plots.scatter(shap_values[:,"HOMO"], color=shap_values[:,"sim_pc"])
            shap.plots.scatter(shap_values[:,"sfe_water"], color=shap_values[:,"sim_pc"])
            shap.plots.scatter(shap_values[:,"sfe_water"], color=shap_values)
            shap.plots.scatter(shap_values[:,"sfe_octa"], color=shap_values)
            shap.plots.scatter(shap_values[:,"neg_partial"], color=shap_values)
            shap.plots.scatter(shap_values[:,"pos_partial"], color=shap_values)
            shap.plots.scatter(shap_values[:,"mag_quad YY"], color=shap_values)
            shap.plots.scatter(shap_values[:,"LUMO"])
            shap.plots.scatter(shap_values[:,"num_benzene"])
            shap.plots.scatter(shap_values[:,"LUMO"], color=shap_values)
            shap.plots.scatter(shap_values[:,"num_benzene"], color=shap_values)
        elif transform == 'YES' and compound == 'YES':
            shap.plots.scatter(shap_values[:,"e^(sfe_octa)*e^(sfe_water)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(sim_pc)*e^(sfe_water)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(sfe_octa)*e^(sfe_water)"], color=shap_values[:,"e^(sim_pc)*e^(sfe_water)"])
            shap.plots.scatter(shap_values[:,"e^(num_o)*(HOMO+1)^1/2"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(neg_partial)*e^(sfe_water)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(mag_quad ZZ)*e^(mag_quad YY)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(mag_quad YY)*e^(mag_quad XX)"], color=shap_values)
            shap.plots.scatter(shap_values[:,"mag_quad YZ*mag_quad XX"])
            shap.plots.scatter(shap_values[:,"e^(neg_partial)*(mag_quad YY+1)^1/2"])
            shap.plots.scatter(shap_values[:,"e^(mag_dipole)*ent^2"], color=shap_values)
            shap.plots.scatter(shap_values[:,"e^(num_o)*mag_quad XZ^2"], color=shap_values)



        with open('shap_values.values' + model+'.txt', 'w') as filehandle:
            for listitem in shap_values.values:
                filehandle.write(f'{listitem}\n')

        with open('shap_values.base_values' + model+'.txt', 'w') as filehandle:
            for listitem in shap_values.base_values:
                filehandle.write(f'{listitem}\n')

        with open('shap_values.data' + model+'.txt', 'w') as filehandle:
            for listitem in shap_values.data:
                filehandle.write(f'{listitem}\n')
################################
    if bootstrap == 'YES':
        print("********Ensemble Recaller Activated********")
        print("Avg. External Test R2: "+str(mean(r2_external_test)))
        print("Avg. External Test MSE: "+str(mean(mse_external_test)))
        print("Avg. External Test MAE: "+str(mean(mae_external_test)))
        print(' ')
    else:
        print("********Recaller Activated********")
        print("External Test R2: "+str(mean(r2_external_test)))
        print("External Test MSE: "+str(mean(mse_external_test)))
        print(' ')


    return X_external_test, y_external_test
###################################
###################################
###################################

    
def plotter(X_train, X_test, y_train, y_test):
    clf = pickle.load(open('models/' + model+'.sav', 'rb'))
    n = 2
    if early_stop == True:
        n = 3
    plt.subplot(1, n, 1)
    plt.scatter(clf.predict(X_train), y_train)
    plt.annotate("R2 = {:.3f}".format(r2_score(y_train, clf.predict(X_train))), xy=(.1, .8), xycoords='axes fraction')
    plt.annotate("MSE = {:.3f}".format(mean_squared_error(y_train, clf.predict(X_train))), xy=(.1, .7), xycoords='axes fraction')
    plt.annotate("RMSE = {:.3f}".format(mean_squared_error(y_train, clf.predict(X_train), squared=False)), xy=(.1, .6), xycoords='axes fraction')
    plt.xlabel("Partition Coefficient - Experiment")
    plt.ylabel("Partition Coefficient - " + model)
    plt.axline((0, 0), slope=1, color='k', transform=plt.gca().transAxes)
    plt.title('Training Set', size = 14, fontweight='bold')
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((0,6,0,6))
    
    plt.subplot(1, n, 2)
    plt.scatter(clf.predict(X_test), y_test)
    plt.annotate("R2 = {:.3f}".format(r2_score(y_test, clf.predict(X_test))), xy=(.1, .8), xycoords='axes fraction')
    plt.annotate("MSE = {:.3f}".format(mean_squared_error(y_test, clf.predict(X_test))), xy=(.1, .7), xycoords='axes fraction')
    plt.annotate("RMSE = {:.3f}".format(mean_squared_error(y_test, clf.predict(X_test), squared=False)), xy=(.1, .6), xycoords='axes fraction')
    plt.xlabel("Partition Coefficient - Experiment")
    plt.ylabel("Partition Coefficient - " + model)
    plt.axline((0, 0), slope=1, color='k', transform=plt.gca().transAxes)
    plt.title('Testing Set', size = 14, fontweight='bold')
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((0,6,0,6))

    if early_stop == True:
        plt.subplot(1, n, 3)
        plt.plot(clf.loss_curve_, label = "Training")
        plt.plot(clf.validation_scores_, label = "Validation")
        plt.legend()

    plt.show()
    
def plotter(X_train, X_test, y_train, y_test):
    clf = pickle.load(open('models/' + model+'.sav', 'rb'))
    n = 2
    if early_stop == True:
        n = 3
    plt.subplot(1, n, 1)
    plt.scatter(clf.predict(X_train), y_train)
    plt.annotate("R2 = {:.3f}".format(r2_score(y_train, clf.predict(X_train))), xy=(.1, .8), xycoords='axes fraction')
    plt.annotate("MSE = {:.3f}".format(mean_squared_error(y_train, clf.predict(X_train))), xy=(.1, .7), xycoords='axes fraction')
    plt.annotate("RMSE = {:.3f}".format(mean_squared_error(y_train, clf.predict(X_train), squared=False)), xy=(.1, .6), xycoords='axes fraction')
    plt.xlabel("Partition Coefficient - Experiment")
    plt.ylabel("Partition Coefficient - " + model)
    plt.axline((0, 0), slope=1, color='k', transform=plt.gca().transAxes)
    plt.title('Training Set', size = 14, fontweight='bold')
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((0,6,0,6))
    
    plt.subplot(1, n, 2)
    plt.scatter(clf.predict(X_test), y_test)
    plt.annotate("R2 = {:.3f}".format(r2_score(y_test, clf.predict(X_test))), xy=(.1, .8), xycoords='axes fraction')
    plt.annotate("MSE = {:.3f}".format(mean_squared_error(y_test, clf.predict(X_test))), xy=(.1, .7), xycoords='axes fraction')
    plt.annotate("RMSE = {:.3f}".format(mean_squared_error(y_test, clf.predict(X_test), squared=False)), xy=(.1, .6), xycoords='axes fraction')
    plt.xlabel("Partition Coefficient - Experiment")
    plt.ylabel("Partition Coefficient - " + model)
    plt.axline((0, 0), slope=1, color='k', transform=plt.gca().transAxes)
    plt.title('Testing Set', size = 14, fontweight='bold')
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((0,6,0,6))

    if early_stop == True:
        plt.subplot(1, n, 3)
        plt.plot(clf.loss_curve_, label = "Training")
        plt.plot(clf.validation_scores_, label = "Validation")
        plt.legend()

    plt.show()

def filter(clf, X_train, bs_imp):
    filtered_features = []
    count = 0
    if clf == 'NULL':
        a = bs_imp['Average Importance']
        a = a.sort_index(ascending=True)
        #b = bs_imp['Feature']
    else:
        a = clf.best_estimator_.coef_
    #print(a)
    for coef in a:
        #print(coef)
        if (not coef == 0.0):
            #filtered_featur = b[count]
            #print(filtered_featur)
            filtered_features.append(X_train.columns[count])
            #filtered_features.append(X_train[filtered_featur])
            #print(filtered_features)
        count+=1
    return filtered_features

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if early_stop == False:
    val_fraction = 0

if not model == 'ANN':
    search = 'GS'

##if lasso_filter_boot == 'YES' and transform == 'YES':
##    "Not advisable or really possible to transform features and bootstrap LASSO!"
##    "If you want to transform features, LASSO bootstrapping will be turned off"
##    lasso_filter_boot = 'NO'
##    lasso_filter = 'YES'
if lasso_filter_boot == 'YES':
    lasso_filter = 'YES'
    
if lasso_filter == 'YES':
    #lasso_filter = 'YES'
    randomSeed = 'YES'
    filtered_features = main('LASSO', lasso_filter_boot, lasso_n_models,'NULL', run_shap)
    if use_latest_model == 'YES':
        plot = 'NO'
    lasso_filter = 'NO'
    if not transform_features == 'YES':
        data_file = 'data_lassoed.csv'
    print(' ')
    print(' ')
else:
    filtered_features = 'NULL'

if (lasso_filter == 'YES' or lasso_filter_boot == 'YES') and use_latest_model == 'YES':
    data_file = 'data_lassoed.csv'

#print("Test/Train size: " + str(len(y_train))+" / " + str(len(y_test)) + "   (" + str((1 - testsize)) + ":" + str(testsize) + ")")
#print(' ')

if (transform_features == 'YES' and compound == 'YES') and lasso_filter == 'NO':
    filtered_features = []
    with open('filtered_trans_features.txt', 'r') as filehandle:
        for line in filehandle:
            # Remove linebreak which is the last character of the string
            curr_place = line[:-1]
            # Add item to the list
            filtered_features.append(curr_place)
    
  
if bootstrap == 'NO':
    #if use_latest_model == 'YES':
    #    n_models = pd.read_csv("test_indices_"+model+".csv").shape[1]
    #else:
    n_models = 1
    print('######################## INITIATING ' + model + ' MODEL ########################')
elif bootstrap == 'YES':
    #if use_latest_model == 'YES':
    #    n_models = pd.read_csv("test_indices_"+model+"_"+str(n_models)+"_random_runs.csv").shape[1]
    randomSeed = 'YES'
    #if use_latest_model == 'YES':
    #    plot = 'NO'
    plot = 'NO'
    print('######################## INITIATING ' + model + ' - ' + str(n_models) + ' MODELS ########################')
    print(' ')
main(model, bootstrap, n_models,filtered_features, run_shap)


