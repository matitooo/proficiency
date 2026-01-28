import torch 
import torch_geometric.nn as tg
import torch.nn as nn
import torch.nn.functional as F


class PopulationGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        """
        Population GCN model
        """
        super(PopulationGCN, self).__init__()
        self.conv1 = tg.GCNConv(input_dim,hidden_dim)
        self.conv2 = tg.GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim,out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        """
        Takes data object as input and extracts features and edges from it. 
        Returns raw probabilities.
        """
        x, edge_index, edge_weight = data.x.to(self.device), data.edge_index.to(self.device), data.edge_weight.to(self.device)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.fc(x)
        return x
    
    def predict(self,data):
        out = self.forward(data)
        pred = torch.argmax(out,dim=1)
        return pred
        
