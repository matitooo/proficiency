import yaml 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from itertools import product
from sklearn.preprocessing import LabelEncoder,StandardScaler


def params_extraction():
    graph_config_path =  "config/graph_config.yaml"
    model_config_path = "config/model_config.yaml"
    df_config_path = "config/df_config.yaml"
    with open(graph_config_path, "r") as f:
        graph_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(df_config_path, "r") as f:
        df_config = yaml.load(f, Loader=yaml.SafeLoader)
    params =df_config | graph_config | model_config
    return params

def data_preprocessing(model_name,params,dataset):
    dataset = dataset.copy(deep=True)
    if model_name == 'linear':  
        columns = params['columns']
        columns_categorical = columns['categorical']
        columns_numerical = columns['numerical']
        columns_binary = columns['binary']
        label = columns['label']
        dataset = dataset.drop(columns=['pseudo','sense','Student_text','Vocab_range'])
        dataset = pd.get_dummies(dataset, columns=columns_categorical+columns_binary)
        # dataset[columns_numerical] = (dataset[columns_numerical] - dataset[columns_numerical].min()) / (dataset[columns_numerical].max() - dataset[columns_numerical].min())
        y = dataset[label].values.ravel()
        le = LabelEncoder()
        y = le.fit_transform(y)
        X = dataset.drop(columns=label)
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
        scaler = StandardScaler()
        X_train[columns_numerical] = scaler.fit_transform(X_train[columns_numerical])
        X_test[columns_numerical] = scaler.transform(X_test[columns_numerical])
        return X_train,X_test,y_train,y_test
    
    elif model_name == 'graph':
        #graph creation 
        columns = params['columns']
        columns_categorical = columns['categorical']
        columns_numerical = columns['numerical']
        columns_binary = columns['binary']
        graph_columns = params['graph_columns']
        threshold = params['threshold']
        n_examples = dataset.shape[0]
        adjacency = np.zeros(shape=(n_examples,n_examples))
        n_features = len(graph_columns)
        for column in graph_columns:
            values = dataset[column].values
            if column in columns_categorical or column in columns_binary:
                adjacency += (values[:, None] == values[None, :]).astype(int)
            elif column in columns_numerical:
                adjacency += 1/(1+(abs(values[:,None] - values[None,:])))
        np.fill_diagonal(adjacency,0)
        adjacency = adjacency/n_features
        adjacency[adjacency<threshold] = 0
    
        adj_tensor = torch.tensor(adjacency,dtype=torch.float)
        edge_index = adj_tensor.nonzero(as_tuple=False).t().contiguous()  
        edge_weight = adj_tensor[adj_tensor != 0]  

        
        #Senses embedding
        
        with open('config/senses_config.yaml', "r") as f:
            sense_dict = yaml.load(f, Loader=yaml.SafeLoader)
        
        senses_dataset = dataset['sense'].values
        senses_sequence = [torch.tensor([sense_dict[sense] for sense in sense_seq],dtype=torch.long) for sense_seq in senses_dataset]
        padded = pad_sequence(senses_sequence, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(s) for s in senses_sequence])
        
        #Label Extraction
        columns = params['columns']
        label_column = dataset['CEFR_level']
        y_values = dataset['CEFR_level'].unique()
        labels_dict = dict(zip(y_values,range(1,len(y_values)+1)))
        label_column = label_column.map(labels_dict)
        y_tensor = torch.tensor(label_column.values, dtype=torch.long)
        
        N = y_tensor.size(0)

        indices = np.arange(N)

        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.3,
            stratify=y_tensor.numpy(),  
            random_state=42
        )

        train_mask = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)

        train_mask[train_idx] = True
        test_mask[test_idx] = True
        data = Data(
            x=padded,
            y=y_tensor,
            lengths=lengths,
            edge_index=edge_index,
            edge_weight=edge_weight,
            train_mask=train_mask,
            test_mask=test_mask
        )

        return data


def generate_model_sweeps(param_grids):
    sweeps = {}

    for model_name, params in param_grids.items():
        keys = list(params.keys())
        values = list(params.values())

        model_configs = []
        for combination in product(*values):
            config = dict(zip(keys, combination))
            model_configs.append(config)

        sweeps[model_name] = model_configs

    return sweeps



