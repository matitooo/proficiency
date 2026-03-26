import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.nn.utils.rnn import pack_padded_sequence

class BiLSTM_GAT_FC(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_layers, gat_hidden_size, num_classes, gat_heads=4):
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
        self.gat2 = GATConv(gat_hidden_size*gat_heads, gat_hidden_size, heads=1, concat=True)
        
        self.relu = nn.ReLU()
        
        # 🔹 FC finale per predizione
        self.fc = nn.Linear(gat_hidden_size, num_classes)
        
    def forward(self, data):
        x, lengths = data.x, data.lengths
        
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.bilstm(packed)
        
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h = torch.cat([h_forward, h_backward], dim=-1)  # (N, lstm_hidden_size*2)
        
        h = self.gat1(h, data.edge_index)
        h = self.relu(h)
        h = self.gat2(h, data.edge_index)
        h = self.relu(h)
        
        out = self.fc(h)  # (N, num_classes)
        return out
    

class MHAttention_GAT(nn.Module):
    def __init__(self, input_size, attn_hidden_size, gat_hidden_size, num_classes, num_heads=4, gat_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, attn_hidden_size)
        self.mha1 = nn.MultiheadAttention(embed_dim=attn_hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim=attn_hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(attn_hidden_size)
        self.norm2 = nn.LayerNorm(attn_hidden_size)
        self.gat1 = GATConv(attn_hidden_size, gat_hidden_size, heads=gat_heads, concat=True)
        self.gat2 = GATConv(gat_hidden_size*gat_heads, gat_hidden_size, heads=1, concat=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(gat_hidden_size, num_classes)
        
    def forward(self, data):
        x, lengths = data.x, data.lengths
        N, L, _ = x.size()
        max_len = x.size(1)
        device = x.device
        mask = torch.arange(max_len, device=device).expand(N, max_len) >= lengths.unsqueeze(1)
        x = self.input_proj(x)
        attn_out1, _ = self.mha1(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out1)
        attn_out2, _ = self.mha2(x, x, x, key_padding_mask=mask)
        x = self.norm2(x + attn_out2)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(2))
        node_emb = x.gather(1, idx).squeeze(1)
        h = self.gat1(node_emb, data.edge_index)
        h = self.relu(h)
        h = self.gat2(h, data.edge_index)
        h = self.relu(h)
        out = self.fc(h)
        return out