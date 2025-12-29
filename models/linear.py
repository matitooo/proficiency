import torch 
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(LinearModel, self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,output_size)
        self.leaky_relu = nn.LeakyReLU()
    
    def forward(self,x):
        x = self.l1(x)
        x = self.leaky_relu(x)
        x = self.l2(x)
        return x
    
    def predict(self,x):
        with torch.no_grad():
            out = self.forward(x)
            pred = torch.argmax(out, dim=1)
        return pred