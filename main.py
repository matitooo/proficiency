from utils import column_dict,normalize_df,graph_creation,one_hot_df,create_tr_te_mask,tensor_extraction
import yaml
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import torch.functional as F
import torch.nn as nn
import torch
from models.linear import LinearModel
from train_utils import train
from test_utils import evaluate

with open("config/graph_config.yaml", "r") as f:
    graph_config = yaml.load(f, Loader=yaml.SafeLoader)

with open("config/df_config.yaml", "r") as f:
    df_config = yaml.load(f, Loader=yaml.SafeLoader)
    
with open("config/model_config.yaml", "r") as f:
    model_config = yaml.load(f, Loader=yaml.SafeLoader)
    
with open("config/train_config.yaml", "r") as f:
    train_config = yaml.load(f, Loader=yaml.SafeLoader)

graph_columns = graph_config["graph_columns"]
threshold = graph_config["threshold"]
df_columns = df_config['columns']
label = df_columns['label']

data_path = "data/celva_transl.csv"

df = pd.read_csv(data_path)
df = df.drop(columns=['L2'])
df = df.dropna(subset=label)

num_classes = len(df[label[0]].unique())

train_mask, test_mask = create_tr_te_mask(df,test_size = 0.2)

graph_columns_map = column_dict(graph_columns, df_columns)
normalized_df = normalize_df(df,df_columns,train_mask)
df_oh = one_hot_df(normalized_df,df_columns)

texts = df_oh['Student_text']
df_oh = df_oh.drop(columns='Student_text')
df_oh[df_oh.columns] = df_oh[df_oh.columns].astype(float)
adjacency,weights = graph_creation(normalized_df,graph_columns_map,threshold)


X,y,train_mask_tensor,test_mask_tensor = tensor_extraction(df_oh,df_columns,train_mask,test_mask)

input_size = X.shape[1]
hidden_size = model_config['hidden_size']
output_size = num_classes

model = LinearModel(input_size,hidden_size,output_size)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr = train_config['lr'])

train(model,optim,loss,train_config['n_epochs'],X,y,train_mask_tensor)
train_acc,train_f1,test_acc,test_f1 = evaluate(model,X,y,train_mask_tensor,test_mask_tensor)