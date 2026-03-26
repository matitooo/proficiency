import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


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

    def forward(self, x,x_lenghts, src_key_padding_mask=None):
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