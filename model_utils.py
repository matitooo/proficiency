from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,accuracy_score
from models.population_gcn import PopulationGCN
import torch
import torch.nn as nn
import torch.optim as optim

def train(model_name,data,params):
    if model_name == 'linear':
        X_train,X_test,y_train,y_test = data
        
        model = MLPClassifier(
        hidden_layer_sizes=(params['hidden_size'],), 
        activation='relu',                  
        solver='adam', 
        learning_rate='constant', 
        learning_rate_init=params['lr'],                    
        max_iter=params['n_epochs'],
        alpha = params['weight_decay'],                      
        random_state=42)
        
        model.fit(X_train,y_train.squeeze())
        return model 
    
    elif model_name == 'graph':
        num_categories = 12           
        embed_dim = 16
        hidden_dim = params['hidden_size']
        out_dim = torch.max(data.y)
        model = PopulationGCN(
            num_categories=num_categories, 
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim+1
        )     
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])  
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        data = data.to(device)
        
        model.train()
        num_epochs = params['n_epochs']

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        
        out = model(data)

        
        loss = criterion(
            out[data.train_mask],
            data.y[data.train_mask]
        )

        
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

    return model


        
    
def test(model_name,model,data):
    if model_name == 'linear':
        X_train,X_test,y_train,y_test = data
        preds = model.predict(X_test)
        acc = accuracy_score(y_test,preds)
        f1 = accuracy_score(y_test,preds)
        print(f"Accuracy : {acc:.3f} F1 : {f1:.3f}")
        return acc,f1
    
    elif model_name == 'graph':
        model.eval()
        with torch.no_grad():
            logits = model(data)             
            preds = logits.argmax(dim=1)       

            y_test = data.y[data.test_mask].cpu().numpy()
            preds = preds[data.test_mask].cpu().numpy()

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            print(f"Accuracy : {acc:.3f} F1 : {f1:.3f}")
        return acc,f1