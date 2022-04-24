import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging


def sample_characteristics(df, path_out, table_type):
    table = pd.DataFrame(index=df.columns)
    table['Unique'] = df.nunique()
    table['Missing'] = 100 * df.isnull().sum() / df.shape[0]
    table['Missing'] = table['Missing'].round(decimals=1)
    table['Min'] = df.min().round(decimals=2)
    table['Max'] = df.max().round(decimals=2)
    table['Mean'] = df.mean().round(decimals=2)
    table['Std Dev'] = df.std().round(decimals=2)
    table.drop(columns=['Min', 'Max'], inplace=True)
    if table_type == 'latex':
        table.to_latex(path_out + '/sample_table.txt')
    elif table_type == 'csv':
        table.to_csv(path_out + '/sample_table.csv')
    else:
        logging.warning(f"table_type must be csv or latex, got: {table_type}")
    return


def feature_plots(df, cols_hist, cols_bar, target, target_label, path_out):
    for col in cols_hist:
        fig, ax = plt.subplots()
        bins = np.histogram(np.hstack((df.loc[df[target] == 0, col].dropna(), df.loc[df[target] == 1, col].dropna())),
                            bins=40)[1]
        ax.hist(df.loc[df[target] == 0, col], bins, alpha=0.5, density=True, stacked=True)
        ax.hist(df.loc[df[target] == 1, col], bins, alpha=0.5, density=True, stacked=True)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_title(col)
        ax.legend(['Control', target_label])
        fig.savefig(path_out + '/' + col + '.png')
        plt.close()
    for col in cols_bar:
        labels = [i for i in range(round(min(df[col])), round(max(df[col])))]
        df_cases = df.loc[df[target] == 1, :]
        df_control = df.loc[df[target] == 0, :]
        df1 = pd.DataFrame(index=labels)
        df1 = pd.concat([df1, df_control[col].value_counts() / df_control.shape[0]], axis=1).rename(
            columns={col: 'Control'})
        df1 = pd.concat([df1, df_cases[col].value_counts() / df_cases.shape[0]], axis=1).rename(
            columns={col: target_label})
        fig, ax = plt.subplots()
        df1.plot(kind='bar', ax=ax, rot=0)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(col)
        plt.tight_layout()
        fig.savefig(path_out + '/' + col + '.png')
        plt.close()
    return


def execute(config):
    path = os.getcwd()
    data_folder = config['general']['data_folder']
    data = pd.read_csv(path + '/data/' + data_folder + '/raw.csv')
    path_out = path + '/output/eda/' + data_folder
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    sample_characteristics(data, path_out, config['eda']['table_type'])
    cols_hist = [col for col in data.columns if col not in config['general']['cat_cols']]
    cols_bar = config['general']['cat_cols']
    feature_plots(data, cols_hist, cols_bar, config['general']['target'], config['general']['target_label'], path_out)
    return
