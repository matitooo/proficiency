import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch_geometric.nn as tg
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PopulationGCN(nn.Module):
    def __init__(self, num_categories, embed_dim, lstm_hidden_size,gcn_hidden_size,dropout, out_dim):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=num_categories + 1,  
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            num_layers= 1,
            bidirectional= False
        )

        # GCN
        self.conv1 = tg.GCNConv(lstm_hidden_size, gcn_hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = tg.GCNConv(gcn_hidden_size, gcn_hidden_size)

        # Classifier
        self.fc = nn.Linear(gcn_hidden_size, out_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):

        x = data.x.to(self.device)  
        edge_index = data.edge_index.to(self.device)
        edge_weight = data.edge_weight.to(self.device)
        lengths = data.lengths.to(self.device)

        x = self.embedding(x)  

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        x = out.gather(1, idx).squeeze(1)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.fc(x)
        return x

    def predict(self, data):
        return torch.argmax(self.forward(data), dim=1)



class PopulationGAT(nn.Module):
    def __init__(self, num_categories, embed_dim, lstm_hidden_size, gat_hidden_size,gat_heads,dropout,out_dim):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=num_categories + 1,  
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            num_layers= 1,
            bidirectional= True
        )

        # GCN
        self.gat1 = tg.GATConv(lstm_hidden_size*2, gat_hidden_size,heads=gat_heads)
        self.gat2 = tg.GATConv(gat_hidden_size*gat_heads, gat_hidden_size,heads=gat_heads)
        self.dropout = nn.Dropout(p=dropout)

        # Classifier
        self.fc = nn.Linear(gat_hidden_size*gat_heads, out_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):

        x = data.x.to(self.device)  
        edge_index = data.edge_index.to(self.device)
        edge_weight = data.edge_weight.to(self.device)
        lengths = data.lengths.to(self.device)

        x = self.embedding(x)  

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        x = out.gather(1, idx).squeeze(1)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        return self.fc(x)

    def predict(self, data):
        return torch.argmax(self.forward(data), dim=1)