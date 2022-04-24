import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from helpers.helper import round_decimals
import logging
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest


def impute_forest(df, cat_cols, random_state):
    imputer = MissForest(verbose=0, criterion=('squared_error', 'gini'), random_state=random_state)
    df_impute = imputer.fit_transform(df, cat_vars=cat_cols)
    return df_impute


def split_impute(df, target, test_size, cat_cols, random_state, round_dict, path, data_folder):
    x = df.drop(columns=[target])
    y = df.loc[x.index, target]
    cat_cols_index = list(df.columns.get_indexer(cat_cols))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    if x.isna().sum().sum() > 0:
        logging.info("Imputing missing values")
        x_train = pd.DataFrame(impute_forest(x_train, cat_cols_index, random_state), columns=x.columns)
        x_test = pd.DataFrame(impute_forest(x_test, cat_cols_index, random_state), columns=x.columns)
    else:
        x_train.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    if len(round_dict) > 0:
        train = round_decimals(train, round_dict)
        test = round_decimals(test, round_dict)
    path_out = path + '/data/' + data_folder
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    train.to_csv(path_out + '/train.csv', index=False)
    test.to_csv(path_out + '/test.csv', index=False)
    return


def execute(config):
    path = os.getcwd()
    data_folder = config['general']['data_folder']
    data = pd.read_csv(path + '/data/' + data_folder + '/raw.csv')
    split_impute(df=data, target=config['general']['target'], test_size=config['preprocessing']['test_size'],
                 cat_cols=config['general']['cat_cols'], random_state=config['general']['random_state'],
                 round_dict=eval(config['general']['round']), path=path, data_folder=data_folder)
    return
