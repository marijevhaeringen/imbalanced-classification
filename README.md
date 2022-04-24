# imbalanced-classification
This programme compares the performance of various generative models, cost-sensitive algorithms and hybrid approaches on binary disease classification problems.
Minority class data can be generated using Synthetic Minority Oversampling TEchnique (SMOTE) or with a Tabular Variational Auto-Encoder (TVAE). 
TVAE models are constructed using a customised version of the [CTGAN](https://pypi.org/project/ctgan/) package. 
There are five options for the predictive models: Random Forest (RF), XGBoost with cross-entropy loss (XGB) and three cost-sensitive XGBoost models.
The cost-sensitive XGBoost models are implemented using a customised version of the [imbalance-xgboost](https://pypi.org/project/imbalance-xgboost/) package.
In addition, this script can perform exploratory data analysis (EDA) and allows hyperparameter tuning for the generative and predictive models.
The Diabetes Health Indicators (DHI) dataset with the objective to classify Diabetes_binary is available. 

## Description
The analyses are performed by running the executer file, where arguments can be specified in the corresponding configuration file config.json. 
The raw dataset should be a csv file (raw.csv) in the following folder: "~/data/[data folder]". 
The name of the data folder, target column, target label (for EDA plots), categorical column names, and possible rounding after data generation are to be specified in the config file. 
There are five tasks: EDA, preprocessing, hyperparameter tuning for the generative models, training the generative models, and training and evaluation the predictive models

### EDA
Creates distribution plots of the features grouped by the target variable and a summary of the characteristics of the variables, which can be in latex or csv format. 

### Preprocessing
Splits the raw data in a train and test set using a stratified split. 
The size of the test set can be adjusted via the config. 

### Generative parameters 
Evaluates the generative models for different hyperparameter values. 
For SMOTE, this includes the number of nearest neighbours, and for TVAE, the network structure of the encoder and decoder, latent dimension and batch size. 
The following quality metrics are available: Kolmogorov-Smirnov (KS) D statistic, Chi-Square test p-value, pairwise correlation difference, log-cluster metric and train loss (only for TVAE).

### Generative models
Fits the generative models to the training data in order to create synthetic minority samples. 
For SMOTE, the number of nearest neighbours can be defined, and for TVAE, the network structure of the encoder and decoder, latent dimension and batch size. 
There is the possibility to calculate the following quality metrics: Kolmogorov-Smirnov (KS) D statistic, Chi-Square test p-value, pairwise correlation difference, log-cluster metric and train loss (only for TVAE).

### Predictive models
Performs randomised search cross-validation to select the optimal hyperparameteres, fits the classifiers to the training data in order to predict the target and evaluates the predictive performance of the fitted models.
Outputs the cross-validation results, F1 and F5 score on the test set, predicted probabilities of the test set, feature importance on the train and test set, and ROC analysis. 
Allows to train a selection of generative models and classifiers.
The predictive models can be compared using the McNemar's test, with exact or approximated p-values. 


