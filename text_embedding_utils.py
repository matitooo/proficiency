import pandas as pd
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import gensim.downloader

glove_vectors = gensim.downloader.load('glove-twitter-25')
glove_vocab = set(glove_vectors.key_to_index)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_sense_labels(words, jsonObj):
    args_order = [[] for _ in range(len(words))]
    senses = jsonObj['relations'][0]
    for sense in senses:
        tokenlist_arg1 = sense['Arg1']['TokenList']
        tokenlist_arg2 = sense['Arg2']['TokenList']
        connective = sense['Connective']['TokenList']
        for token in tokenlist_arg1:
            args_order[token].append(sense['Sense'][0] + '_arg1')
        for token in tokenlist_arg2:
            args_order[token].append(sense['Sense'][0] + '_arg2')
        for token in connective:
            args_order[token].append(sense['Sense'][0] + '_connective')
    return args_order

def create_sense_labels():
    sense_labels_obj = {}
    for file_name in os.listdir('proficiency/data/senses'):    
        file_path = 'proficiency/data/senses/' + file_name
        jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
        num_sentences = len(jsonObj['sentences'][0])
        words = []
        for i in range(num_sentences):
            tokens = jsonObj['sentences'][0][i]['tokens']
            for token in tokens:
                words.append(token['surface'])
        sense_labels = transform_sense_labels(words, jsonObj)
        sense_labels_obj[file_name.split('.')[0]] = sense_labels
    with open('sense_labels.json', 'w') as f:
        f.write(json.dumps(sense_labels_obj))

def create_vocab(sense_labels_dict):
    vocab = set()
    for _id, labels in sense_labels_dict.items():
        vocab =  vocab | set([label for word_labels in labels for label in word_labels])
    return sorted(list(vocab))

def transform_sense_multihot(labels, vocab):
    idx = {tok: i for i, tok in enumerate(vocab)}
    rows = []
    cols = []
    for i, sub in enumerate(labels):
        for tok in set(sub):
            rows.append(i)
            cols.append(idx[tok])
    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)
    multihot = torch.zeros((len(labels), len(vocab)), dtype=torch.float32)
    multihot[rows, cols] = 1.0
    return multihot

def generate_word_embeddings(word_list):
    words_lower = [w.lower() for w in word_list]
    embeddings = np.zeros((len(words_lower), glove_vectors.vector_size), dtype=np.float32)
    for i, word in enumerate(words_lower):
        if word in glove_vocab:
            embeddings[i] = glove_vectors[word]
    return embeddings

def generate_word_list(file_name):
    file_path = './data/senses/' + file_name
    jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
    num_sentences = len(jsonObj['sentences'][0])
    words = []
    for i in range(num_sentences):
        tokens = jsonObj['sentences'][0][i]['tokens']
        for token in tokens:
            words.append(token['surface'])
    return words

def create_data():
    grade_to_int = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
    multihot_dict = {}
    with open('sense_labels.json', 'r') as f:
        sense_labels_dict = json.load(f)
        vocab = create_vocab(sense_labels_dict)
        for id, labels in sense_labels_dict.items():
            multihot_dict[id] = transform_sense_multihot(labels, vocab)
    word_embeddings_dict = {}
    for file_name in os.listdir('./data/senses'):
        id = file_name.split('.')[0]
        words = generate_word_list(file_name)
        word_embeddings_dict[id] = generate_word_embeddings(words)
    y = {}
    df = pd.read_csv('./data/celva_18_23.csv')
    for ind, row in df.iterrows():
        if row['CECRL'] not in grade_to_int.keys():
            y[row['pseudo']] = torch.tensor(-1, dtype=torch.int64)
        else:
            y[row['pseudo']] = torch.tensor(grade_to_int[row['CECRL']], dtype=torch.int64)
    data = []
    for id, multihot in multihot_dict.items():
        concatenated = torch.cat((torch.tensor(word_embeddings_dict[id], dtype=torch.float32), multihot), dim=1)
        if y[id] != -1:
            data.append([id, concatenated, y[id]])
    return data

class SenseDataset(Dataset):
    def __init__(self):
        self.data = create_data()
    def __getitem__(self, key):
        return self.data[key]
    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    tensors = []
    labels = []
    lengths = []
    for instance in batch:
        tensors.append(instance[1])
        lengths.append(len(instance[1]))
        labels.append(instance[2])
    padded_batch = pad_sequence(tensors, batch_first=True)
    return (padded_batch, torch.tensor(labels), lengths)