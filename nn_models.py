import pandas as pd
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import gensim.downloader
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import random_split

glove_vectors = gensim.downloader.load('glove-twitter-25')
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
    for file_name in os.listdir('./data/senses'):    
        file_path = './data/senses/' + file_name
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
    vocab = sorted(list(vocab))
    return vocab

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
    embeddings = []
    for word in word_list:
        if word not in glove_vectors:
            embeddings.append(np.zeros(25))
        else:
            embeddings.append(glove_vectors[word.lower()])
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
    grade_to_int = {
        'A1': 0,
        'A2': 1,
        'B1': 2,
        'B2': 3,
        'C1': 4,
        'C2': 5
    }
    multihot_dict = {}
    with open('sense_labels.json', 'r') as f:
        sense_labels_dict = json.load(f)
        vocab = create_vocab(sense_labels_dict)
        for id, labels in sense_labels_dict.items():
            multihot = transform_sense_multihot(labels, vocab)
            multihot_dict[id] = multihot
    word_embeddings_dict = {}
    for file_name in os.listdir('./data/senses'):
        id = file_name.split('.')[0]
        words = generate_word_list(file_name)
        word_embeddings = generate_word_embeddings(words)
        word_embeddings_dict[id] = word_embeddings
    y = {}
    df = pd.read_csv('./data/celva_18_23.csv')
    for ind, row in df.iterrows():
        if row['CECRL'] not in grade_to_int.keys():
            # mark instances without CECRL level with -1
            y[row['pseudo']] = torch.tensor(-1, dtype=torch.int64)
        else:
            y[row['pseudo']] = torch.tensor(grade_to_int[row['CECRL']], dtype=torch.int64)

    data = []
    for id, multihot in multihot_dict.items():
        concatenated = torch.cat((torch.tensor(word_embeddings_dict[id], dtype=torch.float32), multihot), dim=1)
        if y[id] != -1:
            data.append([id, concatenated, y[id]])
    return data

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, x_lengths):
        x_packed = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(x_packed)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        out = self.fc(h_cat)
        return out

class MHAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.mha1 = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.mha2 = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        attn_out1, _ = self.mha1(
            x, x, x,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + attn_out1)
        attn_out2, _ = self.mha2(
            x, x, x,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm2(x + attn_out2)
        last_step = x[:, -1, :]
        logits = self.fc(last_step)
        return logits

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

create_sense_labels()
dataset = SenseDataset()
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# model = BiLSTM(input_size=60, hidden_size=32, num_layers=2, num_classes=6)
model = MHAttention(input_size=60, hidden_size=32, num_classes=6)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for x, y, x_lengths in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        # outputs = model(x, x_lengths)

        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}", end="\r")

y_preds = np.empty(0)
ys = np.empty(0)
for i, (x, y, x_lengths) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    y_logits = model(x)
    # y_logits = model(x, x_lengths)
    y_pred = torch.argmax(y_logits, dim=1)
    if i == 0:
        y_preds = y_pred.cpu().numpy()
        ys = y.cpu().numpy()
    else:
        y_preds = np.concatenate((y_preds, y_pred.cpu().numpy()))
        ys = np.concatenate((ys, y.cpu().numpy()))
scores = {
    'f1_micro': f1_score(ys, y_preds, average='micro'),
    'f1_macro': f1_score(ys, y_preds, average='macro'),
    'f1_weighted': f1_score(ys, y_preds, average='weighted'),
    'accuracy': accuracy_score(ys, y_preds)
}
print(scores)