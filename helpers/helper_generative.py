import pandas as pd
import numpy as np
from sdv.metrics.tabular import KSTest, CSTest
from sklearn.cluster import KMeans
from helpers.helper import round_decimals
from imblearn.over_sampling import SMOTENC
from CTGAN_custom.ctgan.synthesizers.tvae import TVAESynthesizer
import matplotlib.pyplot as plt
import logging
import warnings


def log_cluster_metric(syn, real, n_clusters, target, random_state):
    real = real.copy()
    syn = syn.copy()
    real['type'] = 'real'
    syn['type'] = 'syn'
    if target in syn.columns:
        syn.drop(columns=[target], inplace=True)
    if target in real.columns:
        real.drop(columns=[target], inplace=True)
    combine = pd.concat([real, syn])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(combine.drop(columns=['type']))
    combine['cluster'] = kmeans.labels_
    c = real.shape[0]/(real.shape[0]+syn.shape[0])
    cluster_sum = []
    for cl in range(n_clusters):
        r = combine.loc[combine['cluster'] == cl, 'type'].str.count('real').sum()
        n = combine.loc[combine['cluster'] == cl, 'type'].shape[0]
        temp = ((r/n) - c)**2
        cluster_sum.append(temp)
    metric = np.log(np.mean(cluster_sum))
    return metric


def eval_syn_data(syn, real, target, cat_cols, n_clusters, random_state):
    result = {}
    syn = syn.copy()
    real = real.copy()
    if target in syn.columns:
        syn.drop(columns=[target], inplace=True)
    if target in real.columns:
        real.drop(columns=[target], inplace=True)
    cont_cols = [col for col in real.columns if col not in cat_cols]
    for col in cat_cols:
        syn[col] = syn[col].astype("object")
        real[col] = real[col].astype("object")
    if len(cont_cols) > 1:
        corr_real = np.triu(np.corrcoef(real[cont_cols], rowvar=False), k=1)
        corr_syn = np.triu(np.corrcoef(syn[cont_cols], rowvar=False), k=1)
        result['l2'] = np.sum(np.power((corr_real - corr_syn), 2))
    else:
        logging.warning(f"Pairwise correlations could not be computed: {len(cont_cols)} continuous variables detected.")
    if len(cat_cols) > 0:
        result['cs'] = CSTest.compute(real[cat_cols], syn[cat_cols])
    else:
        logging.warning(f"Chi-Squared test p-value could not be computed: "
                        f"{len(cat_cols)} categorical variables detected.")
    if len(cont_cols) > 0:
        result['ks'] = KSTest.compute(real[cont_cols], syn[cont_cols])
    else:
        logging.warning(f"Inverted Kolmogorov-Smirnov D-statistic could not be computed: "
                        f"{len(cat_cols)} continuous variables detected.")
    result['log_cluster'] = log_cluster_metric(syn, real, n_clusters, target, random_state)
    return pd.DataFrame(result, columns=result.keys(), index=[0])


def generate_smote(train, test, target, cat_cols, param, round_dict, quality_check, n_clusters, random_state):
    cat_cols_index = list(train.columns.get_indexer(cat_cols))
    x_train = train.drop(columns=[target])
    y_train = train[target]
    sm = SMOTENC(random_state=random_state, k_neighbors=param['k_neighbors'], categorical_features=cat_cols_index)
    x_smote, y_smote = sm.fit_resample(x_train, y_train)
    syn_data = x_smote.drop(index=x_train.index)
    syn_data[target] = y_smote.drop(index=x_train.index)
    syn_data = round_decimals(syn_data, round_dict)
    if quality_check:
        test_cases = test.loc[test[target] == 1, :]
        result = eval_syn_data(syn=syn_data, real=test_cases, target=target, cat_cols=cat_cols, n_clusters=n_clusters,
                               random_state=random_state)
    return syn_data, result if quality_check else syn_data


def generate_tvae(train, test, target, cat_cols, param, round_dict, quality_check, n_clusters, random_state):
    x_train = train.drop(columns=[target])
    y_train = train[target]
    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    for col in cat_cols:
        x_train[col] = x_train[col].astype("object")
    x_train = x_train.loc[y_train == 1, :]
    sample_size = y_train.shape[0] - 2*x_train.shape[0]
    model = TVAESynthesizer(batch_size=param['batch_size'], embedding_dim=param['embedding_dim'],
                            decompress_dims=param['decompress_dims'],
                            compress_dims=param['compress_dims'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, discrete_columns=cat_cols)
    syn_data = model.sample(sample_size)
    syn_data[target] = 1
    syn_data = round_decimals(syn_data, round_dict)
    if quality_check:
        test_cases = test.loc[test[target] == 1, :]
        result = eval_syn_data(syn=syn_data, real=test_cases, target=target, cat_cols=cat_cols, n_clusters=n_clusters,
                               random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result['trainloss'] = model.evaluate(x_train, batch=20)
    return syn_data, result if quality_check else syn_data


def hyperparam_plots(result, params, metrics, color_map, title_map, params_label, metrics_label, model_label, path_out):
    for c_x in params:
        for c_y in metrics:
            fig, ax = plt.subplots()
            mean = result[[c_x, c_y]].groupby(by=[c_x]).mean()
            q90 = result[[c_x, c_y]].groupby(by=[c_x]).quantile(0.9)
            q10 = result[[c_x, c_y]].groupby(by=[c_x]).quantile(0.1)
            asym_error = [mean[c_y].values - q10[c_y].values, q90[c_y].values - mean[c_y].values]
            if c_x == 'compress_dims':
                res = result[c_x].apply(lambda x: str(x))
                ax.errorbar(res.unique(), mean[c_y].values, fmt='o', yerr=asym_error, color=color_map[c_y])
            else:
                ax.errorbar(mean.index, mean[c_y].values, fmt='o', yerr=asym_error, color=color_map[c_y])
            ax.set_title(title_map[c_y])
            if c_x == 'compress_dims':
                plt.xticks(rotation=30)
            ax.set_xlabel(params_label[c_x])
            ax.set_ylabel(metrics_label[c_y])
            plt.tight_layout()
            fig.savefig(path_out + '/' + model_label + '_' + c_x + '_' + c_y + '.png')
            plt.close()
