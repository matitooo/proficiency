from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from models.population_gcn import PopulationGCN,PopulationGAT
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train(model_type,data,params):
    
    if model_type == 'linear':
        X_train,X_test,y_train,y_test = data
        y_train = y_train.astype(np.int64) 
        linear_model = params['model_name']
        
        if linear_model=='MLP':
                    model = MLPClassifier(
                    hidden_layer_sizes=(params['hidden_size'],),
                    activation='relu',
                    solver='adam',
                    learning_rate='constant',
                    learning_rate_init=params['lr'],
                    max_iter=params['n_epochs'],
                    alpha=params['weight_decay'],
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,   
                    n_iter_no_change=10,       
                    tol=1e-4                   
                )
        elif linear_model == 'DecisionTree':
            model = DecisionTreeClassifier(
                max_depth= params['max_depth'],
                min_samples_leaf= params['min_samples_leaf'],
                min_samples_split = params['min_samples_split'],
                criterion = params['criterion']
            )
        
        elif linear_model == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators = params['n_estimators'],
                max_depth = params['max_depth'],
                min_samples_leaf = params['min_samples_leaf'],
                min_samples_split = params['min_samples_split'],
                max_features = params['max_features']
            )
        
        elif linear_model == 'Logreg':
            model = LogisticRegression(
                C = params['C'],
                penalty=params['penalty'],
                solver=params['solver'],
                max_iter=params['max_iter']
            )
        model.fit(X_train,y_train.squeeze())
        return model
    
    elif model_type == 'graph':
        model_name = params['model_name']
        num_categories = 12           
        embed_dim = params['embed_dim']
        hidden_dim = params['hidden_size']
        out_dim = torch.max(data.y)
        if model_name == 'GCN':
            model = PopulationGCN(
                num_categories=num_categories, 
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim+1
            )     
        elif model_name == 'GAT':
            model = PopulationGAT(
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
    prev_loss = 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        
        out = model(data)
        loss = criterion(
            out[data.train_mask],
            data.y[data.train_mask]
        )
        print(abs(prev_loss-loss.item()))
        if abs(prev_loss-loss.item()) < 0.005:
            break 
        else:
            prev_loss = loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

    return model


        
    
def test(model_type,model,data):
    if model_type == 'linear':
        X_train,X_test,y_train,y_test = data
        model_type = type(model).__name__
        preds = model.predict(X_test)
        acc = accuracy_score(y_test,preds)
        f1 = f1_score(y_test,preds,average='macro')
        print(f"Model Name : {model_type} Accuracy : {acc:.3f} F1 : {f1:.3f}")
        return acc,f1
    
    elif model_type == 'graph':
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