from itertools import product
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
import optuna
from text_embedding_utils import SenseDataset, collate_fn
from model_utils import train, test

sense_dataset = SenseDataset()

def params_extraction():
    "Extracts graph/model/dataset params and creates parameters dictionary"
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
    "Preprocess data based on model type, returns data object"
    dataset = dataset.copy(deep=True)
    # Linear Models
    if model_name == 'linear':  
        columns = params['columns']
        columns_categorical = columns['categorical']
        columns_numerical = columns['numerical']
        columns_binary = columns['binary']
        label = columns['label']
        dataset = dataset.drop(columns=['pseudo','sense','Student_text','Vocab_range'])
        dataset = pd.get_dummies(dataset, columns=columns_categorical+columns_binary)
        y = dataset[label].values.ravel()
        le = LabelEncoder()
        y = le.fit_transform(y)
        X = dataset.drop(columns=label)
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
        scaler = StandardScaler()
        X_train[columns_numerical] = scaler.fit_transform(X_train[columns_numerical])
        X_test[columns_numerical] = scaler.transform(X_test[columns_numerical])
        return X_train,X_test,y_train,y_test
    
    # Sequential Models
    elif model_name =='sequential':
        train_dataset, test_dataset = random_split(sense_dataset, [0.7, 0.3])
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
        return [train_loader,test_loader]
    
    # Mixed Models
    elif model_name == 'mixed':
        data_list = sense_dataset.data   

        id_to_tensor = {instance[0]: instance[1] for instance in data_list}
        id_to_label  = {instance[0]: instance[2] for instance in data_list}

        ordered_ids = dataset['pseudo'].values

        tensors = []
        labels  = []

        for i in ordered_ids:
            tensors.append(id_to_tensor[i])
            labels.append(id_to_label[i])

        padded = pad_sequence(tensors, batch_first=True)  
        lengths = torch.tensor([t.shape[0] for t in tensors])
        y_tensor = torch.stack(labels)

        columns = params['columns']
        columns_categorical = columns['categorical']
        columns_numerical   = columns['numerical']
        columns_binary      = columns['binary']

        graph_columns = params['graph_columns']
        threshold = params['threshold']

        n_examples = len(tensors)
        adjacency = np.zeros((n_examples, n_examples))
        n_features = len(graph_columns)

        for column in graph_columns:
            values = dataset[column].values
            if column in columns_categorical or column in columns_binary:
                adjacency += (values[:, None] == values[None, :]).astype(float)
            elif column in columns_numerical:
                adjacency += 1 / (1 + np.abs(values[:, None] - values[None, :]))

        np.fill_diagonal(adjacency, 0)
        adjacency = adjacency / n_features
        adjacency[adjacency < threshold] = 0

        adj_tensor = torch.tensor(adjacency, dtype=torch.float)
        edge_index = adj_tensor.nonzero(as_tuple=False).t().contiguous()
        edge_weight = adj_tensor[adj_tensor != 0]

        N = y_tensor.size(0)
        indices = np.arange(N)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.3
        )

        train_mask = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        data = Data(
            x=padded,            
            lengths=lengths,      
            y=y_tensor,
            edge_index=edge_index,
            edge_weight=edge_weight,
            train_mask=train_mask,
            test_mask=test_mask
        )

        return data
    
    # Graph Models
    elif model_name == 'graph':
        columns = params['columns']
        columns_categorical = columns['categorical']
        columns_numerical = columns['numerical']
        columns_binary = columns['binary']
        graph_columns = params['graph_columns']
        threshold = params['threshold']
        n_examples = dataset.shape[0]
        n_features = len(graph_columns)

        adjacency = np.zeros((n_examples, n_examples))
        for column in graph_columns:
            values = dataset[column].values
            if column in columns_categorical or column in columns_binary:
                adjacency += (values[:, None] == values[None, :]).astype(float)
            elif column in columns_numerical:
                adjacency += 1 / (1 + np.abs(values[:, None] - values[None, :]))
        np.fill_diagonal(adjacency, 0)
        adjacency = adjacency / n_features
        adjacency[adjacency < threshold] = 0

        adj_tensor = torch.tensor(adjacency, dtype=torch.float)
        edge_index = adj_tensor.nonzero(as_tuple=False).t().contiguous()
        edge_weight = adj_tensor[adj_tensor != 0]

        with open('config/senses_config.yaml', "r") as f:
            sense_dict = yaml.load(f, Loader=yaml.SafeLoader)

        senses_dataset = dataset['sense'].values
        senses_sequence = [
            torch.tensor([sense_dict[sense] for sense in sense_seq], dtype=torch.long)
            for sense_seq in senses_dataset
        ]
        padding_idx = len(sense_dict)
        padded = pad_sequence(senses_sequence, batch_first=True, padding_value=padding_idx)
        lengths = torch.tensor([len(s) for s in senses_sequence])

        cefr_dict = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
        label_column = dataset['CEFR_level'].map(cefr_dict)
        y_tensor = torch.tensor(label_column.values, dtype=torch.long)
        N = y_tensor.size(0)

        indices = np.arange(N)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.3)

        train_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
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


def sweep_params_gen(model_name):
    "Stores Sweep Parameters. Return sweep parameters for selected model"
    if model_name == 'MLP':
        sweep = {
            'lr': [0.001, 0.005, 0.01],
            'n_epochs': [50, 100],
            'hidden_size': [64, 128, 256],
            'weight_decay': [0.0, 0.005, 0.01],
            'activation': ['relu', 'tanh']
        }

    if model_name == 'DecisionTree':
        sweep = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'criterion': ['gini', 'entropy']
        }

    if model_name == 'RandomForest':
        sweep = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2', None]
        }

    if model_name == 'Logreg':
        sweep = {
            'C': [0.1, 1.0, 5.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['saga', 'liblinear'],
            'max_iter': [500, 1000, 2000]
        }

    if model_name == 'GCN':
        sweep = {
            'gcn_hidden_size': [64, 128, 256],
            'lstm_hidden_size': [64, 128, 256],
            'embed_dim': [16, 32,64],
            'dropout': [0.0,0.1,0.3],
            'lr': [0.001, 0.005, 0.01],
            'weight_decay':[0.01,0.005],
            'n_epochs': [20, 50, 100,200],
            'threshold' : [0.5,0.7,0.9],
            'graph_columns': [
    "Years_studying_L2|Reinforced_section",
    "Years_studying_L2|Language_exposure",
    "Years_studying_L2|Reading_frequency",
    "Reinforced_section|Language_exposure",
    "Reinforced_section|Reading_frequency",
    "Language_exposure|Reading_frequency",

    "Years_studying_L2|Reinforced_section|Language_exposure",
    "Years_studying_L2|Reinforced_section|Reading_frequency",
    "Years_studying_L2|Language_exposure|Reading_frequency",
    "Reinforced_section|Language_exposure|Reading_frequency",

    "Years_studying_L2|Reinforced_section|Language_exposure|Reading_frequency"
]
        }

    if model_name == 'GAT':
        sweep = {
            'embed_dim': [16, 32, 64],
            'lstm_hidden_size': [64,128,256],
            'gat_hidden_size' : [64,128,256],
            'gat_heads' : [1,2,4],
            'dropout': [0.0,0.1,0.3],
            'lr': [0.001, 0.005, 0.01],
            'weight_decay':[0.01,0.005],
            'n_epochs': [20, 50, 100],
            'threshold' : [0.5,0.7,0.9],
            'graph_columns': [
    "Years_studying_L2|Reinforced_section",
    "Years_studying_L2|Language_exposure",
    "Years_studying_L2|Reading_frequency",
    "Reinforced_section|Language_exposure",
    "Reinforced_section|Reading_frequency",
    "Language_exposure|Reading_frequency",

    "Years_studying_L2|Reinforced_section|Language_exposure",
    "Years_studying_L2|Reinforced_section|Reading_frequency",
    "Years_studying_L2|Language_exposure|Reading_frequency",
    "Reinforced_section|Language_exposure|Reading_frequency",

    "Years_studying_L2|Reinforced_section|Language_exposure|Reading_frequency"
]
            
        }

    if model_name == 'BiLSTM':
        sweep = {
            'input_size': [60],
            'num_classes': [6],
            'lstm_hidden_size': [64,128,256],
            'lr': [0.001, 0.005, 0.01],
            'n_epochs': [20, 50, 100,200],
            'weight_decay':[0.01,0.005]
        }

    if model_name == 'MHAttention':
        sweep = {
            'input_size': [60],
            'hidden_size': [32, 64, 128,256],
            'num_heads' : [1,2,4],
            'num_classes': [6],
            'lr': [0.001, 0.005, 0.01],
            'n_epochs': [10, 20, 50],
            'weight_decay':[0.01,0.005]
        }

    if model_name == 'BiLSTM_GAT_FC':
       sweep = {
    'input_size': [60],
    'lstm_hidden_size': [64,128,256],
    'gat_hidden_size' : [64,128,256],
    'lstm_layers':[1,2],
    'embed_dim': [16, 32, 64],
    'gat_heads' : [1,2,4],
    'dropout': [0.0,0.1,0.3],
    'num_classes': [6],
    'lr': [0.001, 0.005, 0.01],
    'weight_decay':[0.01,0.005],
    'n_epochs': [20, 50, 100],
    'threshold' : [0.5,0.7,0.9],
    'graph_columns': [
    "Years_studying_L2|Reinforced_section",
    "Years_studying_L2|Language_exposure",
    "Years_studying_L2|Reading_frequency",
    "Reinforced_section|Language_exposure",
    "Reinforced_section|Reading_frequency",
    "Language_exposure|Reading_frequency",

    "Years_studying_L2|Reinforced_section|Language_exposure",
    "Years_studying_L2|Reinforced_section|Reading_frequency",
    "Years_studying_L2|Language_exposure|Reading_frequency",
    "Reinforced_section|Language_exposure|Reading_frequency",

    "Years_studying_L2|Reinforced_section|Language_exposure|Reading_frequency"
]
}
    
    if model_name == 'MHAttention_GAT_FC':
        sweep  = {'input_size': [60],
    'embed_dim' : [64,128,256],
    'num_classes': [6],
    'gat_hidden_size' : [64,128,256],
    'attn_heads': [1,2,4],
    'gat_heads' : [1,2,4],
    'dropout': [0.0,0.1,0.3],
    'lr': [0.001, 0.005, 0.01],
    'weight_decay':[0.01,0.005],
    'n_epochs': [20, 50, 100],
    'threshold' : [0.5,0.7,0.9],
    'graph_columns': [
    "Years_studying_L2|Reinforced_section",
    "Years_studying_L2|Language_exposure",
    "Years_studying_L2|Reading_frequency",
    "Reinforced_section|Language_exposure",
    "Reinforced_section|Reading_frequency",
    "Language_exposure|Reading_frequency",

    "Years_studying_L2|Reinforced_section|Language_exposure",
    "Years_studying_L2|Reinforced_section|Reading_frequency",
    "Years_studying_L2|Language_exposure|Reading_frequency",
    "Reinforced_section|Language_exposure|Reading_frequency",

    "Years_studying_L2|Reinforced_section|Language_exposure|Reading_frequency"
]}
        
    return sweep


def create_study_for_model(model_type,dataset,model_name,sweep_params):
    "Creates optuna study for bayesian hyperparameter tuning for selected model"
    def objective(trial):
        params = params_extraction()
        for param, values in sweep_params.items():
            params[param] = trial.suggest_categorical(param, values)
        if 'graph_columns' in sweep_params.keys():
            params['graph_columns'] = params['graph_columns'].split("|")
            print(params['graph_columns'])
        data_copy = dataset.copy(deep=True)
        data = data_preprocessing(model_type, params, data_copy)
        params['model_name'] = model_name
        
        trained_model = train(model_type,data,params)
        scores = test(model_type,trained_model,data)
        return scores['f1_micro']

    study = optuna.create_study(direction='maximize', study_name=f"{model_name}_study")
    return study, objective