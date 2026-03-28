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


class MHAttention_GAT_FC(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        num_classes,
        gat_hidden_size,
        attn_heads=4,
        gat_heads=4,
        dropout=0.1
    ):
        super().__init__()

        # 🔹 Input projection
        self.input_proj = nn.Linear(input_size, embedding_size)

        # 🔹 Multi-head attention blocks
        self.mha1 = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True
        )
        self.mha2 = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        # 🔹 GAT layers
        self.gat1 = GATConv(embedding_size, gat_hidden_size, heads=gat_heads, concat=True)
        self.gat2 = GATConv(gat_hidden_size * gat_heads, gat_hidden_size, heads=gat_heads, concat=True)

        self.norm_gat1 = nn.LayerNorm(gat_hidden_size * gat_heads)
        self.norm_gat2 = nn.LayerNorm(gat_hidden_size * gat_heads)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 🔹 Final classifier
        self.fc = nn.Linear(gat_hidden_size * gat_heads, num_classes)

    def forward(self, data):
        x, lengths = data.x, data.lengths
        device = x.device

        # 🔹 Build padding mask (True = PAD)
        max_len = x.size(1)
        mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]

        # 🔹 Attention encoder
        x = self.input_proj(x)

        attn_out1, _ = self.mha1(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out1)

        attn_out2, _ = self.mha2(x, x, x, key_padding_mask=mask)
        x = self.norm2(x + attn_out2)

        # 🔹 Mean pooling (mask-aware)
        valid_mask = (~mask).unsqueeze(-1)  # (N, T, 1)
        x = x * valid_mask

        sum_x = x.sum(dim=1)
        lengths = lengths.clamp(min=1).unsqueeze(1)  # avoid division by zero
        h = sum_x / lengths  # (N, hidden_size)

        # 🔹 GAT
        h = self.gat1(h, data.edge_index)
        h = self.norm_gat1(h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.gat2(h, data.edge_index)
        h = self.norm_gat2(h)
        h = self.relu(h)
        h = self.dropout(h)

        # 🔹 Classification
        out = self.fc(h)
        return out