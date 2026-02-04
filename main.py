from utils import params_extraction,data_preprocessing
from model_utils import train,test
import pandas as pd
import ast


params = params_extraction(sweep=False)

data_path = 'data/data.csv'
dataset = pd.read_csv(data_path,index_col = 0)
dataset["Other_language"] = dataset["Other_language"].fillna("Aucune")
dataset["Reading_frequency"] = dataset["Reading_frequency"].fillna(0)




dataset = dataset.dropna()

dataset["sense"] = dataset["sense"].apply(ast.literal_eval)
dataset = dataset[dataset['sense'].apply(len) > 1]
#model = 'linear'
model = 'graph'

mode = 'train'
# mode = 'sweep'

if mode == 'train':
    data = data_preprocessing(model,params,dataset)
    trained_model = train(model,data,params)
    test(model,trained_model,data)
    



