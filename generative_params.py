import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
from CTGAN_custom.ctgan.synthesizers.tvae import TVAESynthesizer
from helpers.helper import round_decimals
from helpers.helper_generative import generate_smote, generate_tvae, hyperparam_plots
import logging


def hyperparam_smote(train, val, target, cat_cols, param, round_dict, n_clusters, random_state, path_out, color_map,
                     title_map, params_label, metrics_label):
    tune_results = pd.DataFrame()
    logging.info(f"Fitting SMOTE models with different hyperparameter values.")
    for k in param['k_neighbors']:
        syn_data, result = generate_smote(train=train, test=val, target=target, cat_cols=cat_cols,
                                          param={'k_neighbors': k}, round_dict=round_dict, quality_check=True,
                                          n_clusters=n_clusters, random_state=random_state)
        metrics = result.columns
        result['k_neighbors'] = k
        tune_results = pd.concat([tune_results, result], axis=0).reset_index(drop=True)
    tune_results.to_csv(path_out + '/hyperparam_smote.csv', index=False)
    hyperparam_plots(result=tune_results, params=param.keys(), metrics=metrics, color_map=color_map,
                     title_map=title_map, params_label=params_label, metrics_label=metrics_label, model_label='smote',
                     path_out=path_out)
    return


def hyperparam_tvae(train, val, target, cat_cols, param, round_dict, n_clusters, random_state, path_out, color_map,
                    title_map, params_label, metrics_label):
    tune_results = pd.DataFrame()
    logging.info(f"Fitting TVAE models with different hyperparameter values.")
    for em in param['embedding_dim']:
        for c in param['compress_dims']:
            for b in param['batch_size']:
                temp_param = {'embedding_dim': em, 'compress_dims': c, 'decompress_dims': c, 'batch_size': b}
                syn_data, result = generate_tvae(train=train, test=val, target=target, cat_cols=cat_cols,
                                                 param=temp_param, round_dict=round_dict, quality_check=True,
                                                 n_clusters=n_clusters, random_state=random_state)
                metrics = result.columns
                result['embedding_dim'] = em
                result['compress_dims'] = str(c)
                result['batch_size'] = b
                tune_results = pd.concat([tune_results, result]).reset_index(drop=True)
    tune_results.to_csv(path_out + '/hyperparam_tvae.csv', index=False)
    hyperparam_plots(result=tune_results, params=param.keys(), metrics=metrics, color_map=color_map,
                     title_map=title_map, params_label=params_label, metrics_label=metrics_label, model_label='tvae',
                     path_out=path_out)
    return


def execute(config):
    path = os.getcwd()
    train_data = pd.read_csv(path + '/data/' + config['general']['data_folder'] + '/train.csv')
    train, val = train_test_split(train_data, test_size=config['generative_params']['test_size'],
                                  random_state=config['general']['random_state'],
                                  stratify=train_data[config['general']['target']])
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    path_out = path + '/output/generative params/' + config['general']['data_folder']
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    if config['generative_params']['smote']:
        hyperparam_smote(train, val, target=config['general']['target'], cat_cols=config['general']['cat_cols'],
                         param=eval(config['generative_params']['smote_param']),
                         round_dict=eval(config['general']['round']),
                         n_clusters=config['generative_params']['n_clusters'],
                         random_state=config['general']['random_state'], path_out=path_out,
                         color_map=config['generative_params']['color_map'],
                         title_map=config['generative_params']['title_map'],
                         params_label=config['generative_params']['param_label_smote'],
                         metrics_label=config['generative_params']['metrics_label'])
    if config['generative_params']['tvae']:
        hyperparam_tvae(train, val, target=config['general']['target'], cat_cols=config['general']['cat_cols'],
                        param=eval(config['generative_params']['tvae_param']),
                        round_dict=eval(config['general']['round']),
                        n_clusters=config['generative_params']['n_clusters'],
                        random_state=config['general']['random_state'], path_out=path_out,
                        color_map=config['generative_params']['color_map'],
                        title_map=config['generative_params']['title_map'],
                        params_label=config['generative_params']['param_label_tvae'],
                        metrics_label=config['generative_params']['metrics_label'])
    return
