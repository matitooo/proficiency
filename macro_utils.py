from micro_utils import create_tr_te_mask,normalize_df,one_hot_df,column_dict,graph_creation,adjacency_to_pyg,tensor_extraction
import yaml
import pandas as pd
from torch_geometric.data import Data

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


def data_processing(data_path,params):
    df_columns = params['columns']
    graph_columns = params['graph_columns']
    
    label = df_columns['label']
    threshold = params['threshold']
    
    df = pd.read_csv(data_path)
    df = df.dropna(subset=label)
    num_classes = len(df[label[0]].unique())

    train_mask, test_mask = create_tr_te_mask(df,test_size = 0.2)

    normalized_df = normalize_df(df,df_columns,train_mask)
    df_oh = one_hot_df(normalized_df,df_columns)

    texts = df_oh['Student_text']
    df_oh = df_oh.drop(columns='Student_text')
    df_oh[df_oh.columns] = df_oh[df_oh.columns].astype(float)

    graph_columns_map = column_dict(graph_columns, df_columns)
    adjacency,weights = graph_creation(normalized_df,graph_columns_map,threshold)
    edge_index, edge_weight = adjacency_to_pyg(adjacency, weights)

    X,y,train_mask_tensor,test_mask_tensor = tensor_extraction(df_oh,df_columns,train_mask,test_mask)

    data = Data(
        x=X,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        train_mask=train_mask_tensor,
        test_mask=test_mask_tensor
    )
    
    data_params = {"X":X,"y":y,"train_mask":train_mask_tensor,"test_mask":test_mask,"n_classes":num_classes,"data":data}
    return data_params


def model_params_processing(params):
    model_params = {'hidden_size':params['hidden_size']}    
    return model_params

def train_params_processing(params):
    if 'weight_decay' in params.keys():
        weight_decay = params['weight_decay']
    else:
        weight_decay = 0
    train_params = {'lr' : params['lr'], 'n_epochs':params['n_epochs'],'weight_decay' : weight_decay}
    return train_params



