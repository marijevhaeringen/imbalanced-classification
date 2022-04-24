import os
import logging
import json
import eda
import preprocessing
import generative_params
import generative_models
import predictive_models

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename='logging.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

path = os.getcwd()
with open(path + '/config.json') as f:
    config = json.load(f)

tasks = [task for task in config['tasks'].keys() if config['tasks'][task]]
logging.info(f"Running data pipeline for {config['general']['data_folder']} dataset with target "
             f"{config['general']['target']}")
logging.info(f"Performing the following {len(tasks)} tasks: {', '.join(tasks)}")

if 'eda' in tasks:
    path_in = path + '/data/' + config['general']['data_folder'] + '/raw.csv'
    assert os.path.isfile(path_in), f"Raw data file not found. Expected file location: {path_in}"
    logging.info(f"Start exploratory data analysis")
    eda.execute(config)
    logging.info(f"Finished exploratory data analysis")


if 'preprocessing' in tasks:
    path_in = path + '/data/' + config['general']['data_folder'] + '/raw.csv'
    assert os.path.isfile(path_in), f"Raw data file not found. Expected file location: {path_in}"
    logging.info(f"Start preprocessing")
    preprocessing.execute(config)
    logging.info(f"Finished preprocessing")


if 'generative_params' in tasks:
    path_in_train = path + '/data/' + config['general']['data_folder'] + '/train.csv'
    assert os.path.isfile(path_in_train), f"Train data file not found. Expected file location: {path_in_train}"
    logging.info(f"Start evaluating generative models at different hyperparameters")
    generative_params.execute(config)
    logging.info(f"Finished evaluating generative models at different hyperparameters")


if 'generative_models' in tasks:
    path_in_train = path + '/data/' + config['general']['data_folder'] + '/train.csv'
    path_in_test = path + '/data/' + config['general']['data_folder'] + '/test.csv'
    assert os.path.isfile(path_in_train), f"Train data file not found. Expected file location: {path_in_train}"
    assert os.path.isfile(path_in_test), f"Test data file not found. Expected file location: {path_in_test}"
    logging.info(f"Start generating data")
    generative_models.execute(config)
    logging.info(f"Finished generating data")


if 'predictive_models' in tasks:
    path_in_train = path + '/data/' + config['general']['data_folder'] + '/train.csv'
    path_in_test = path + '/data/' + config['general']['data_folder'] + '/test.csv'
    assert os.path.isfile(path_in_train), f"Train data file not found. Expected file location: {path_in_train}"
    assert os.path.isfile(path_in_test), f"Test data file not found. Expected file location: {path_in_test}"
    logging.info(f"Start training and evaluating predictive models")
    predictive_models.execute(config)
    logging.info(f"Finished training and evaluating predictive models")
