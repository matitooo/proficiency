import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch_geometric.nn as tg
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PopulationGCN(nn.Module):
    def __init__(self, num_categories, embed_dim, hidden_dim, out_dim):
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
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers= 2,
            bidirectional= True
        )

        # GCN
        self.conv1 = tg.GCNConv(hidden_dim*2, hidden_dim*2)
        self.conv2 = tg.GCNConv(hidden_dim*2, hidden_dim*2)

        # Classifier
        self.fc = nn.Linear(hidden_dim*2, out_dim)

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
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        return self.fc(x)

    def predict(self, data):
        return torch.argmax(self.forward(data), dim=1)
