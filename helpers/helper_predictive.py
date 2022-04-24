import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, fbeta_score, make_scorer, f1_score
from scipy.stats import randint, uniform
import xgboost
from sklearn.inspection import permutation_importance
from imxgboost_custom.imbalance_xgb import imbalance_xgboost as imb_xgb
from statsmodels.stats import contingency_tables, multitest
import numpy as np
import pickle
import os
import warnings


def get_features_target(df, target):
    x = df.drop(columns=[target])
    y = df[target]
    return x, y


def combine_real_generated(x, y, gen_data, target):
    x_gen = gen_data.drop(columns=[target])
    y_gen = gen_data[target]
    x_combine = pd.concat([x, x_gen]).reset_index(drop=True)
    y_combine = pd.concat([y, y_gen]).reset_index(drop=True)
    return x_combine, y_combine


def roc_performance(model, x_test, y_test, model_label):
    if model_label in ['rf', 'xgb']:
        fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    else:
        fpr, tpr, threshold = roc_curve(y_test, model.predict_sigmoid(x_test))
    roc_values = pd.DataFrame()
    roc_values['sensitivity'] = tpr
    roc_values['specificity'] = 1 - fpr
    roc_values['threshold'] = threshold
    return roc_values


def feature_importance(model, x_train, y_train, x_test, y_test, scoring, path_out, model_label, method_label, random_state):
    importance = permutation_importance(model, x_test, y_test, n_repeats=20, random_state=4, scoring=scoring)
    forest_importances = pd.Series(importance.importances_mean, index=x_test.columns)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=importance.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on test set")
    ax.set_ylabel("Mean F1 decrease")
    fig.tight_layout()
    fig.savefig(path_out + "/feature_importance_test_" + model_label + '_' + method_label + ".png")
    plt.close()
    forest_importances.to_csv(path_out + "/importance_test_" + model_label + '_' + method_label + ".csv")
    importance = permutation_importance(model, x_train, y_train, n_repeats=20, random_state=random_state, scoring=scoring)
    forest_importances = pd.Series(importance.importances_mean, index=x_train.columns)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=importance.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on train set")
    ax.set_ylabel("Mean F1 decrease")
    fig.tight_layout()
    fig.savefig(path_out + "/feature_importance_train_" + model_label + '_' + method_label + ".png")
    plt.close()
    forest_importances.to_csv(path_out + "/importance_train_" + model_label + '_' + method_label + ".csv")
    return


def custom_f1(y_true, y_pred):
    y_pred = np.round(1 / (1 + np.exp(-y_pred)))
    score = f1_score(y_true=y_true, y_pred=y_pred)
    return score


def model_results(clf_model, train_list, test_list, model_label, method_label, path_out, random_state, save_model):
    x_train = train_list[0]
    y_train = train_list[1]
    x_test = test_list[0]
    y_test = test_list[1]
    # cv results
    cv_results = pd.DataFrame(clf_model.cv_results_)
    cv_results.to_csv(path_out + "/cv_results_" + model_label + '_' + method_label + ".csv", index=False)
    # performance: f1, f5 score
    if model_label in ['rf', 'xgb']:
        y_pred = clf_model.best_estimator_.predict(x_test)
        y_proba = clf_model.best_estimator_.predict_proba(x_test)[:, 1]
        score = make_scorer(f1_score)
    else:
        y_pred = clf_model.best_estimator_.predict_determine(x_test)
        y_proba = clf_model.best_estimator_.predict_sigmoid(x_test)
        score = make_scorer(custom_f1)
    performance = pd.Series([fbeta_score(y_true=y_test, y_pred=y_pred, beta=1),
                             fbeta_score(y_true=y_test, y_pred=y_pred, beta=5)])
    performance.to_csv(path_out + "/performance_" + model_label + '_' + method_label + ".csv")
    # roc analysis
    roc_values = roc_performance(clf_model.best_estimator_, x_test, y_test, model_label)
    roc_values.to_csv(path_out + "/roc_" + model_label + '_' + method_label + ".csv", index=False)
    # predictions
    predict = pd.DataFrame()
    predict['true'] = y_test
    predict['predict'] = y_proba
    predict.to_csv(path_out + "/predictions_" + model_label + '_' + method_label + ".csv", index=False)
    # feature importance train & test set
    feature_importance(clf_model, x_train, y_train, x_test, y_test, score, path_out, model_label, method_label,
                       random_state)
    # save model
    if save_model:
        out_saved_models = path_out + "/saved models"
        if not os.path.exists(out_saved_models):
            os.makedirs(out_saved_models)
        pickle.dump(clf_model, open(out_saved_models + "/" + + model_label + '_' + method_label + ".pkl", 'wb'))
    return


def rf(train_list, model_params, n_iter, cv, random_state):
    x_train = train_list[0]
    y_train = train_list[1]
    model = RandomForestClassifier(random_state=random_state)
    clf = RandomizedSearchCV(model, model_params, n_iter=n_iter, cv=cv, random_state=random_state,
                             scoring=make_scorer(f1_score))
    clf_model = clf.fit(x_train, y_train)
    return clf_model


def xgb(train_list, model_params, n_iter, cv, random_state):
    x_train = train_list[0]
    y_train = train_list[1]
    model = xgboost.XGBClassifier(objective='binary:logistic', random_state=random_state)
    clf = RandomizedSearchCV(model, model_params, n_iter=n_iter, cv=cv, random_state=random_state,
                             scoring=make_scorer(f1_score))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_model = clf.fit(x_train.to_numpy(), y_train.to_numpy())
    return clf_model


def xgbf(train_list, model_params, n_iter, cv, random_state):
    x_train = train_list[0]
    y_train = train_list[1]
    model = imb_xgb(special_objective='focal')
    clf = RandomizedSearchCV(model, model_params, n_iter=n_iter, cv=cv, random_state=random_state,
                             scoring=make_scorer(custom_f1))
    clf_model = clf.fit(x_train.to_numpy(), y_train.to_numpy())
    return clf_model


def xgbw(train_list, model_params, n_iter, cv, random_state):
    x_train = train_list[0]
    y_train = train_list[1]
    model = imb_xgb(special_objective='weighted')
    clf = RandomizedSearchCV(model, model_params, n_iter=n_iter, cv=cv, random_state=random_state,
                             scoring=make_scorer(custom_f1))
    clf_model = clf.fit(x_train.to_numpy(), y_train.to_numpy())
    return clf_model


def xgbwf(train_list, model_params, n_iter, cv, random_state):
    x_train = train_list[0]
    y_train = train_list[1]
    model = imb_xgb(special_objective='weighted focal')
    clf = RandomizedSearchCV(model, model_params, n_iter=n_iter, cv=cv, random_state=random_state,
                             scoring=make_scorer(custom_f1))
    clf_model = clf.fit(x_train.to_numpy(), y_train.to_numpy())
    return clf_model


def performance_plot(path_in, score, score_label, path_out):
    files = [f for f in os.listdir(path_in) if 'performance' in f]
    assert len(files) > 0, f"No performance metrics of trained models were found. Expected file location: {path_in}"
    performance = pd.DataFrame(columns=['classifier', 'method', 'f1', 'f5'])
    i = 0
    for f in files:
        performance.loc[i, ['f1', 'f5']] = pd.read_csv(path_in + f, usecols=[1]).transpose().values[0]
        performance.loc[i, 'classifier'] = f.split('_')[1]
        performance.loc[i, 'method'] = f.split('_')[2].split('.')[0]
        i += 1
    performance = performance.sort_values(by=['method', 'classifier'])
    colmap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
    fig, ax = plt.subplots()
    i = 0
    for n in performance['classifier'].unique():
        ax.plot(performance.loc[performance['classifier'] == n, 'method'],
                performance.loc[performance['classifier'] == n, score], 'o-', color=colmap[i])
        i += 1
    ax.set_xlabel('Generative model')
    ax.set_ylabel(score_label)
    ax.set_title(score_label)
    ax.legend(labels=performance['classifier'].unique())
    plt.tight_layout()
    fig.savefig(path_out + '/' + score + '.png')
    plt.close()
    return


def compare_models(path_in, exact, path_out):
    files = [f for f in os.listdir(path_in) if 'predictions' in f]
    files_copy = files.copy()
    files.sort()
    results = pd.DataFrame(columns=['model1', 'model2', 'statistic', 'pvalue'])
    for f1 in files:
        files_copy.remove(f1)
        pred_1 = pd.read_csv(path_in + f1, usecols=[0, 1])
        for f2 in files_copy:
            pred_2 = pd.read_csv(path_in + f2, usecols=[0, 1])
            cont_table = pd.crosstab(pred_1['predict'].round(0) == pred_1['true'],
                                     pred_2['predict'].round(0) == pred_2['true'])
            mc_test = contingency_tables.mcnemar(cont_table, exact=exact)
            temp = dict()
            temp['model1'] = f1.split('_')[1] + '_' + f1.split('_')[2].split('.')[0]
            temp['model2'] = f2.split('_')[1] + '_' + f2.split('_')[2].split('.')[0]
            temp['statistic'] = mc_test.statistic
            temp['pvalue'] = mc_test.pvalue
            results = results.append(temp, ignore_index=True)
    adj_pvalues = multitest.multipletests(results['pvalue'], alpha=0.05, method='fdr_by')
    results['adj pvalue'] = adj_pvalues[1]
    results.to_csv(path_out + '/model_comparison.csv')
    return
