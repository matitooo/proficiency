from utils import params_extraction,data_preprocessing,sweep_params_gen,create_study_for_model
from model_utils import train,test
import pandas as pd
import ast
import yaml
from tqdm import tqdm
from itertools import product
import argparse
import optuna


def train_mode(model_type):
    with open('config/train_config.yaml', "r") as f:
        train_config = yaml.load(f, Loader=yaml.SafeLoader)
    models_dict = train_config[model_type]
    models_list = models_dict.keys()
    out_q = {}
    for model in models_list:
        model_out = []
        for i in range(10):
            print(f"Now training {model}")
            params = params_extraction()
            model_config = models_dict[model]
            for p in model_config.keys():
                params[p] =model_config[p]
            data = data_preprocessing(model_type,params,dataset)
            trained_model = train(model_type,data,params)
            y_test,predictions = test(model_type,trained_model,data,quantitative_flag=True)
            model_out.append[y_test,predictions]
            model_out.append([y_test,predictions])
        out_q[model] = model_out
    rows = []

    for model_name, runs in out_q.items():
        for y_true, y_pred in runs:   # ogni run
            for yt, yp in zip(y_true, y_pred):
                rows.append({
                    "name": model_name,
                    "y_true": yt,
                    "y_pred": yp
                })

    df = pd.DataFrame(rows)
    return df

def quantitative_mode(model_type):
    df = train_mode(model_type)
    name = model_type+'.csv'
    df.to_csv(name)
    
def sweep_mode(model_type):
    with open('config/train_config.yaml', "r") as f:
        train_config = yaml.load(f, Loader=yaml.SafeLoader)
    models_dict = train_config[model_type]
    models_list = models_dict.keys()
    
    
    results = []
    for model in models_list:
        print(f"Now training {model}")
        sweep_params = sweep_params_gen(model)
        study, objective = create_study_for_model(model_type,dataset,model,sweep_params)
        n_trials = 20 if model_type == 'linear' else 200
        study.optimize(objective, n_trials=n_trials) 
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        print(f"Best params for {model}: {best_params} -> Score: {best_value}")

        results.append({
            'model': model,
            'best_score': best_value,
            'params' : best_params
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(f'best_params_{model_type}.csv', index=False)
    print(f"Best parameters saved to best_params_{model_type}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose mode")

    parser.add_argument('--train', action='store_true',
                        help="Train and compare models")
    parser.add_argument('--sweep', action='store_true',
                        help="Find the best Hyperparameters configuration")
    parser.add_argument('--quantitative',action='store_true',help='Retrieve predictions')

    parser.add_argument('--model', type=str, choices=["linear", "graph","sequential","mixed"],
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
        train_mode(model_type = args.model)
    elif args.sweep:
        sweep_mode(model_type = args.model)
    
    if args.quantitative:
        quantitative_mode(model_type = args.model)
        
        
        
        
        
        
