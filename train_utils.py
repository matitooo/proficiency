import torch 

def train(model, optim, loss_f, n_epochs, X, y, train_mask):
    for i in range(n_epochs+1):
        optim.zero_grad()
        out = model(X)  
        loss = loss_f(out[train_mask], y[train_mask]) 
        loss.backward()
        optim.step()
        print(f"Epoch: {i+1} Loss: {loss.item()}")
    return None