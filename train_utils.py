from models.linear import LinearModel
from models.population_gcn import PopulationGCN
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from test_utils import evaluate_linear,evaluate_gcn

def train_linear(model_params, train_params,data_params):
    hidden_size = model_params['hidden_size']
    X = data_params['X']
    y = data_params['y']
    train_mask  = data_params['train_mask']
    test_mask = data_params ['test_mask']
    n_classes = data_params['n_classes']
    learning_rate = train_params['lr']
    weight_decay = train_params['weight_decay']
    n_epochs  = train_params['n_epochs']
    model = LinearModel(X.shape[1],hidden_size,n_classes)
    optim = Adam(params=model.parameters(),lr = learning_rate,weight_decay=weight_decay)
    loss_f = CrossEntropyLoss()
    for i in range(n_epochs+1):
        optim.zero_grad()
        out = model(X)  
        loss = loss_f(out[train_mask], y[train_mask]) 
        loss.backward()
        optim.step()
        if i%20==0:
            print(f"Epoch: {i+1} Loss: {loss.item()}")
    evaluate_linear(model,X,y,train_mask,test_mask)
    return None

def train_gcn(model_params, train_params,data_params):
    hidden_size = model_params['hidden_size']
    data = data_params['data']
    train_mask  = data_params['train_mask']
    test_mask = data_params ['test_mask']
    n_classes = data_params['n_classes']
    learning_rate = train_params['lr']
    weight_decay = train_params['weight_decay']
    n_epochs  = train_params['n_epochs']
    model = PopulationGCN(data.x.shape[1],hidden_size,n_classes)
    optim = Adam(params=model.parameters(),lr = learning_rate,weight_decay=weight_decay)
    loss_f = CrossEntropyLoss()

    for epoch in range(n_epochs):
        optim.zero_grad()

        out = model(data) 

        loss = loss_f(
            out[data.train_mask],
            data.y[data.train_mask]
        )

        loss.backward()
        optim.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
    evaluate_gcn(model,data)