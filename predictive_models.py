import pandas as pd
import os
from helpers.helper_predictive import get_features_target, combine_real_generated, model_results
from scipy.stats import randint, uniform
from helpers.helper_predictive import rf, xgb, xgbf, xgbw, xgbwf, performance_plot, compare_models
import logging


def get_datasets(train, target, smote_bool, tvae_bool, path_in):
    x_train, y_train = get_features_target(train, target)
    train_dict = {'original': [x_train, y_train]}
    if smote_bool:
        path_smote = path_in + '/smote.csv'
        assert os.path.isfile(path_smote), f"prediction with SMOTE was set to true, but no sample was found. " \
                                           f"Expected file at: {path_smote}"
        smote = pd.read_csv(path_smote)
        x_smote, y_smote = combine_real_generated(x_train, y_train, smote, target)
        train_dict['smote'] = [x_smote, y_smote]
    if tvae_bool:
        path_tvae = path_in + '/tvae.csv'
        assert os.path.isfile(path_tvae), f"prediction with TVAE was set to true, but no sample was found. " \
                                          f"Expected file at: {path_tvae}"
        tvae = pd.read_csv(path_tvae)
        x_tvae, y_tvae = combine_real_generated(x_train, y_train, tvae, target)
        train_dict['tvae'] = [x_tvae, y_tvae]
    return train_dict


def execute(config):
    path = os.getcwd()
    path_in = path + '/data/' + config['general']['data_folder']
    train = pd.read_csv(path_in + '/train.csv')
    test = pd.read_csv(path_in + '/test.csv')
    path_out = path + '/output/predictive models/' + config['general']['data_folder']
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    train_dict = get_datasets(train, target=config['general']['target'],
                              smote_bool=config['predictive_models']['smote'],
                              tvae_bool=config['predictive_models']['tvae'], path_in=path_in)
    x_test, y_test = get_features_target(test, config['general']['target'])
    predictors = [pred for pred in ['rf', 'xgb', 'xgbf', 'xgbw', 'xgbwf'] if config['predictive_models'][pred]]
    logging.info(f"Evaluating {len(predictors)} prediction models ({', '.join(predictors)}) on {len(train_dict.keys())} "
                 f"training sets ({', '.join(train_dict.keys())})")
    for pred in predictors:
        for method in train_dict.keys():
            logging.info(f"Performing randomised search CV for {pred} on {method} training set")
            clf_model = eval(pred)(train_list=train_dict[method],
                                   model_params=eval(config['predictive_models'][pred + '_param']),
                                   n_iter=config['predictive_models']['n_iter'], cv=config['predictive_models']['cv'],
                                   random_state=config['general']['random_state'])
            logging.info(f"Calculating performance metrics for the best model for {pred} on {method} training set")
            model_results(clf_model, train_list=train_dict[method], test_list=[x_test, y_test], model_label=pred,
                          method_label=method, path_out=path_out, random_state=config['general']['random_state'],
                          save_model=config['predictive_models']['save_model'])
    if not os.path.exists(path_out + '/evaluate'):
        os.makedirs(path_out + '/evaluate')
    logging.info(f"Evaluating the performance of the predictive models")
    performance_plot(path_in=path_out + '/', score='f1', score_label='F1 score', path_out=path_out + '/evaluate')
    performance_plot(path_in=path_out + '/', score='f5', score_label='F5 score', path_out=path_out + '/evaluate')
    if config['predictive_models']['compare']:
        compare_models(path_in=path_out + '/', exact=config['predictive_models']['exact_pvalues'],
                       path_out=path_out + '/evaluate')
    return
