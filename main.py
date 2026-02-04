from utils import params_extraction,data_preprocessing
from model_utils import train,test
import pandas as pd
import ast
import yaml
from itertools import product

data_path = 'data/data.csv'
dataset = pd.read_csv(data_path,index_col = 0)
dataset["Other_language"] = dataset["Other_language"].fillna("Aucune")
dataset["Reading_frequency"] = dataset["Reading_frequency"].fillna(0)
dataset = dataset.dropna()
dataset["sense"] = dataset["sense"].apply(ast.literal_eval)
dataset = dataset[dataset['sense'].apply(len) > 1]

model = 'linear'
# model = 'graph'

# mode = 'train'
mode = 'sweep'

if mode == 'train':
    params = params_extraction()
    data = data_preprocessing(model,params,dataset)
    trained_model = train(model,data,params)
    test(model,trained_model,data)

elif mode == 'sweep':
    params = params_extraction()
    
    
    with open('config/sweep_config.yaml', "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.SafeLoader)
    param_names = list(sweep_config.keys())
    param_values = list(sweep_config.values())

    combinations = [dict(zip(param_names, values)) for values in product(*param_values)]
    results = {}
    for combination in combinations:
        for param in combination.keys():
            params[param] = combination[param]
        data = data_preprocessing(model,params,dataset)
        trained_model = train(model,data,params)
        acc,f1 = test(model,trained_model,data)
        results[str(combination)] = {"Accuracy":acc,"F1":f1}


