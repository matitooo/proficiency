import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from itertools import product
from torch_geometric.data import Data
import yaml

def create_tr_te_mask(df,test_size = 0.2):
    indices = np.arange(df.shape[0])

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42
    )

    train_mask = np.zeros(df.shape[0], dtype=bool)
    test_mask  = np.zeros(df.shape[0], dtype=bool)

    train_mask[train_idx] = True
    test_mask[test_idx]   = True
    return train_mask,test_mask

def column_dict(columns_considered, df_columns):
    categorical_list = df_columns['categorical']
    numerical_list = df_columns['numerical']
    binary_list = df_columns['binary']
    categ = dict()
    for column in columns_considered:
        if column in categorical_list:
            categ[column] = "categorical"
        elif column in numerical_list:
            categ[column] = "numerical"
        elif column in binary_list:
            categ[column] = "binary"
    return categ


def normalize_df(df,df_columns,train_mask):
    numerical_columns = df_columns['numerical']
    for column in numerical_columns:
        mu = df[column][train_mask].mean()
        sigma = df[column][train_mask].std() + 1e-8
        df[column] = df[column].astype(float)
        df.loc[df[column].isna(), column] = mu
        df[column] = (df[column] - mu) /sigma
    binary_columns = df_columns['binary']
    df[df[column].isna()] = 0
    return df


def one_hot_df(df, df_columns):
    cat_cols = df_columns['categorical']
    label = df_columns['label'][0]
    levels = sorted(df[label].unique())
    label_map = {level: i for i, level in enumerate(levels)}
    df[label] = df[label].map(label_map)
    df_cat = pd.get_dummies(df[cat_cols], dummy_na=False)
    return pd.concat([df.drop(columns=cat_cols), df_cat], axis=1)

def graph_creation(df,gr_columns,threshold):
    n = df.shape[0]
    n_feat = len(gr_columns.keys())
    adjacency = np.zeros(shape=(n,n))
    for column in gr_columns.keys():
        values = df[column].values
        if gr_columns[column] == "categorical" or gr_columns[column] == "binary":
            adjacency += (values[:, None] == values[None, :]).astype(int)
        elif gr_columns[column] == "numerical":
            adjacency += 1/(1+(abs(values[:,None] - values[None,:])))
    np.fill_diagonal(adjacency,0)
    adjacency = adjacency/n_feat
    adjacency[adjacency<threshold] = 0
    weights = adjacency
    adjacency[adjacency!=0] = 1
    return adjacency,weights

def tensor_extraction(df,df_columns,train_mask,test_mask):
    label = df_columns['label']
    y = torch.tensor(df[label[0]].values, dtype=torch.long)
    df = df.drop(columns = label[0])
    X = torch.tensor(df.values, dtype=torch.float32)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    test_mask  = torch.tensor(test_mask, dtype=torch.bool)
    return X,y,train_mask,test_mask


def create_sweep_dict(config_dict):
    combinations = [
        dict(zip(config_dict.keys(), values))
        for values in product(*config_dict.values())
    ]
    return combinations

def create_data_object(
    X,                
    edge_index,       
    edge_weight=None,
    y=None,            
    train_mask=None,
    test_mask=None
):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))

    data = Data(
        x=X,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask
    )

    return data


def adjacency_to_pyg(adjacency, weights):
    src, dst = np.nonzero(adjacency)

    edge_index = torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long
    )

    edge_weight = torch.tensor(
        weights[src, dst],
        dtype=torch.float32
    )

    return edge_index, edge_weight


def params_extraction(sweep = False):
    graph_config_path =  "config/graph_config.yaml"
    model_config_path = "config/model_config.yaml"
    train_config_path = "config/train_config.yaml"
    df_config_path = "config/df_config.yaml"
    if sweep:
        train_config_path = "config/sweep_config.yaml"
    with open(graph_config_path, "r") as f:
        graph_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(train_config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(df_config_path, "r") as f:
        df_config = yaml.load(f, Loader=yaml.SafeLoader)
    # params = {**df_config,**graph_config, **model_config, **train_config}
    params =df_config | graph_config | model_config |train_config
    return params
