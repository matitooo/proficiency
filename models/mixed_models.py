import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.nn.utils.rnn import pack_padded_sequence

class BiLSTM_GAT_FC(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_layers, gat_hidden_size, num_classes,dropout, gat_heads):
        super().__init__()
        
        # 🔹 BiLSTM
        self.bilstm = nn.LSTM(
            input_size,
            lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 🔹 GAT
        self.gat1 = GATConv(lstm_hidden_size*2, gat_hidden_size, heads=gat_heads, concat=True)
        self.gat2 = GATConv(gat_hidden_size*gat_heads, gat_hidden_size, heads=gat_heads, concat=True)
        self.dropout = nn.Dropout(p=dropout)
        
        self.relu = nn.ReLU()
        
        # 🔹 FC finale per predizione
        self.fc = nn.Linear(gat_hidden_size*gat_heads, num_classes)
        
    def forward(self, data):
        print(data.x.shape)
        x, lengths = data.x, data.lengths
        
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.bilstm(packed)
        
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h = torch.cat([h_forward, h_backward], dim=-1)  # (N, lstm_hidden_size*2)
        
        h = self.gat1(h, data.edge_index)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.gat2(h, data.edge_index)
        h = self.relu(h)
        
        out = self.fc(h)  # (N, num_classes)
        return out

