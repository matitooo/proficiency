from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from models.population_gcn import PopulationGCN,PopulationGAT
from models.sequential_models import BiLSTM,MHAttention
from models.mixed_models import BiLSTM_GAT_FC
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model_type,data,params):
    
    if model_type == 'linear':
        X_train,X_test,y_train,y_test = data
        y_train = y_train.astype(np.int64) 
        linear_model = params['model_name']
        
        if linear_model == 'MLP':
            model = MLPClassifier(
                hidden_layer_sizes=(params['hidden_size'],),
                activation=params['activation'],
                solver='adam',
                learning_rate='constant',
                learning_rate_init=params['lr'],
                max_iter=params['n_epochs'],
                alpha=params['weight_decay'],
                batch_size=params['batch_size'],
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4
            )
        elif linear_model == 'DecisionTree':
            model = DecisionTreeClassifier(
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                min_samples_split=params['min_samples_split'],
                max_features=params['max_features'],
                criterion=params['criterion'],
                class_weight=params['class_weight'],
                random_state=42
            )
        
        elif linear_model == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                min_samples_split=params['min_samples_split'],
                max_features=params['max_features'],
                bootstrap=params['bootstrap'],
                class_weight=params['class_weight'],
                random_state=42,
                n_jobs=-1
            )
    
        elif linear_model == 'Logreg':
            model = LogisticRegression(
            C=params['C'],
            penalty=params['penalty'],
            solver=params['solver'],
            max_iter=params['max_iter'],
            class_weight=params['class_weight'],
            fit_intercept=params['fit_intercept'],
            tol=params['tol'],
            random_state=42
        )
        
        model.fit(X_train,y_train.squeeze())
        return model
    
    elif model_type =='sequential':
        train_loader,test_loader = data
        model_name = params['model_name']
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        num_classes = params['num_classes']
        num_epochs = params['n_epochs']
        lr = params['lr']
        if model_name == 'BiLSTM':
            num_layers = params['num_layers']
            model = BiLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
        elif model_name=='MHAttention':
            model = MHAttention(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        avg_loss_old = 0
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            for x, y, x_lengths in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                outputs = model(x,x_lengths)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if abs(avg_loss_old-avg_loss) < 0.0001:
                break 
            else:
                avg_loss_old = avg_loss
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}", end="\r")
        return model
        
    elif model_type == 'mixed':
        model_name= params['model_name']
        input_size = params['input_size']
        lstm_hidden_size = params['lstm_hidden_size']
        lstm_layers = params['lstm_layers']
        gat_hidden_size = params['gat_hidden_size']
        gat_heads = params['gat_heads']
        num_classes = params['num_classes']
        n_epochs = params['n_epochs']
        lr = params['lr']
        if model_name == 'BiLSTM_GAT_FC':
            model = BiLSTM_GAT_FC(input_size, lstm_hidden_size, lstm_layers, gat_hidden_size, num_classes, gat_heads)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            data = data.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr= lr)  
            model.train()
        prev_loss = 0
        for epoch in tqdm(range(num_epochs)):
            model.train()
            optimizer.zero_grad()

            
            out = model(data)
            loss = criterion(
                out[data.train_mask],
                data.y[data.train_mask]
            )
            if abs(prev_loss-loss.item()) < 0.0001:
                break 
            else:
                prev_loss = loss.item()
                loss.backward()
                optimizer.step()
            if epoch%10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

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
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()

        
        out = model(data)
        loss = criterion(
            out[data.train_mask],
            data.y[data.train_mask]
        )
        if abs(prev_loss-loss.item()) < 0.0001:
            break 
        else:
            prev_loss = loss.item()
            loss.backward()
            optimizer.step()
        if epoch%10 == 0:
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
        scores = {'acc':acc,'f1':f1}
        return scores
    
    elif model_type == 'graph':
        model.eval()
        with torch.no_grad():
            logits = model(data)             
            preds = logits.argmax(dim=1)       

            y_test = data.y[data.test_mask].cpu().numpy()
            preds = preds[data.test_mask].cpu().numpy()

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            scores = {'acc':acc,'f1':f1}
        return scores
    
    elif model_type=='sequential':
        train_loader,test_loader = data
        y_preds = np.empty(0)
        ys = np.empty(0)
        for i, (x, y, x_lengths) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            y_logits = model(x,x_lengths)
            y_pred = torch.argmax(y_logits, dim=1)
            if i == 0:
                y_preds = y_pred.cpu().numpy()
                ys = y.cpu().numpy()
            else:
                y_preds = np.concatenate((y_preds, y_pred.cpu().numpy()))
                ys = np.concatenate((ys, y.cpu().numpy()))
        scores = {
            'f1_micro': f1_score(ys, y_preds, average='micro'),
            'f1_macro': f1_score(ys, y_preds, average='macro'),
            'f1_weighted': f1_score(ys, y_preds, average='weighted'),
            'accuracy': accuracy_score(ys, y_preds)
        }

        return scores
  

