import pandas as pd
import os
from helpers.helper_generative import generate_smote, generate_tvae
import logging


def generate_write_smote(train, test, target, cat_cols, param, round_dict, quality_check, n_clusters, random_state,
                         path_out):
    logging.info(f"Generating synthetic data using SMOTE")
    syn_data, *result = generate_smote(train=train, test=test, target=target, cat_cols=cat_cols, param=param,
                                       round_dict=round_dict, quality_check=quality_check, n_clusters=n_clusters,
                                       random_state=random_state)
    if quality_check:
        result[0].to_csv(path_out + '/smote_quality.csv', index=False)
    syn_data.to_csv(path_out + '/smote.csv', index=False)
    return


def generate_write_tvae(train, test, target, cat_cols, param, round_dict, quality_check, n_clusters, random_state,
                        path_out):
    logging.info(f"Generating synthetic data using TVAE")
    syn_data, *result = generate_tvae(train=train, test=test, target=target, cat_cols=cat_cols, param=param,
                                      round_dict=round_dict, quality_check=quality_check, n_clusters=n_clusters,
                                      random_state=random_state)
    if quality_check:
        result[0].to_csv(path_out + '/tvae_quality.csv', index=False)
    syn_data.to_csv(path_out + '/tvae.csv', index=False)
    return


def execute(config):
    path = os.getcwd()
    train = pd.read_csv(path + '/data/' + config['general']['data_folder'] + '/train.csv')
    test = pd.read_csv(path + '/data/' + config['general']['data_folder'] + '/test.csv')
    path_out = path + '/data/' + config['general']['data_folder']
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    if config['generative_models']['smote']:
        generate_write_smote(train=train, test=test, target=config['general']['target'],
                             cat_cols=config['general']['cat_cols'],
                             param=eval(config['generative_models']['smote_param']),
                             round_dict=eval(config['general']['round']),
                             quality_check=config['generative_models']['quality'],
                             n_clusters=config['generative_models']['n_clusters'],
                             random_state=config['general']['random_state'], path_out=path_out)

    if config['generative_models']['tvae']:
        generate_write_tvae(train=train, test=test, target=config['general']['target'],
                            cat_cols=config['general']['cat_cols'],
                            param=eval(config['generative_models']['tvae_param']),
                            round_dict=eval(config['general']['round']),
                            quality_check=config['generative_models']['quality'],
                            n_clusters=config['generative_models']['n_clusters'],
                            random_state=config['general']['random_state'], path_out=path_out)
    return
