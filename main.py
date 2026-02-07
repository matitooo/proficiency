from utils import params_extraction,data_preprocessing,generate_model_sweeps
from model_utils import train,test
import pandas as pd
import ast
import yaml
from tqdm import tqdm
from itertools import product
import argparse





def train_mode(model):
    with open('config/train_config.yaml', "r") as f:
        train_config = yaml.load(f, Loader=yaml.SafeLoader)
    models_list = train_config.keys()
    
    if model == 'linear':
    
        for linear_model in models_list:
            params = params_extraction()
            linear_model_config = train_config[linear_model]
            for p in linear_model_config.keys():
                params[p] = linear_model_config[p]
            data = data_preprocessing(model,params,dataset)
            trained_model = train(model,data,params)
            acc,f1 = test(model,trained_model,data)
    
    elif model == 'graph':
        params = params_extraction()
        linear_model_config = train_config['Graph']
        for p in linear_model_config.keys():
            params[p] = linear_model_config[p]
        data = data_preprocessing(model,params,dataset)
        trained_model = train(model,data,params)
        acc,f1 = test(model,trained_model,data)
        

def sweep_mode(model):
    if model == 'linear':
        params = params_extraction()
        with open('config/sweep_config.yaml', "r") as f:
            sweep_config = yaml.load(f, Loader=yaml.SafeLoader)
            del sweep_config['Graph']
        
        

    elif model == 'graph':
        params = params_extraction()
        with open('config/sweep_config.yaml', "r") as f:
            sweep_config = yaml.load(f, Loader=yaml.SafeLoader)
            sweep_params_graph = sweep_config['Graph']
            sweep_config = dict()
            sweep_config['Graph'] = sweep_params_graph
        
    combinations = generate_model_sweeps(sweep_config)  
    df = pd.DataFrame()
    rows = []
    data = data_preprocessing(model,params,dataset)
    for m, configs in combinations.items():
        print(f"Now tuning {m}")
        for sweep_params in tqdm(configs):
            sweep_params['model_name'] = m
            trained_model = train(model,data,sweep_params)
            acc,f1 = test(model,trained_model,data)
            sweep_params['Test Accuracy'] = acc
            sweep_params['Test F1'] = f1
            row = {"model": m}
            row.update(sweep_params)
            rows.append(row)
        
    df = pd.DataFrame(rows)
    pd.DataFrame.to_csv(df,'results_'+model+'.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose mode")

    parser.add_argument('--train', action='store_true',
                        help="Train and compare models")
    parser.add_argument('--sweep', action='store_true',
                        help="Find the best Hyperparameters configuration")

    parser.add_argument('--model', type=str, choices=["linear", "graph"],
                        required=True,
                        help="Select model type: Linear or Graph")

    args = parser.parse_args()
    data_path = 'data/data.csv'
    dataset = pd.read_csv(data_path,index_col = 0)
    dataset["Other_language"] = dataset["Other_language"].fillna("Aucune")
    dataset["Reading_frequency"] = dataset["Reading_frequency"].fillna(0)
    dataset = dataset.dropna()
    dataset["sense"] = dataset["sense"].apply(ast.literal_eval)
    dataset = dataset[dataset['sense'].apply(len) > 1]
    
    if args.train:
        train_mode(model = args.model)
    elif args.sweep:
        sweep_mode(model = args.model)
        
        
        
        
        
        
        