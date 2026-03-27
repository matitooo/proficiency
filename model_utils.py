from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
from models.population_gcn import PopulationGCN,PopulationGAT
from models.sequential_models import BiLSTM,MHAttention
from models.mixed_models import BiLSTM_GAT_FC
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model_type, data, params):

    # Linear models
    if model_type == 'linear':
        X_train, X_test, y_train, y_test = data
        y_train = y_train.astype(np.int64)
        model_name = params['model_name']

        if model_name == 'MLP':
            model = MLPClassifier(
                hidden_layer_sizes=(params['hidden_size'],),
                activation=params['activation'],
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

        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier(
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                criterion=params['criterion'],
                random_state=42
            )

        elif model_name == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=42,
                n_jobs=-1
            )

        elif model_name == 'Logreg':
            model = LogisticRegression(
                C=params['C'],
                penalty=params['penalty'],
                solver=params['solver'],
                max_iter=params['max_iter'],
                random_state=42
            )

        model.fit(X_train, y_train.squeeze())
        return model

    # Sequential models
    elif model_type == 'sequential':
        train_loader, test_loader = data
        model_name = params['model_name']

        input_size = params['input_size']
        num_classes = params['num_classes']
        num_epochs = params['n_epochs']
        lr = params['lr']
        weight_decay = params['weight_decay']

        if model_name == 'BiLSTM':
            model = BiLSTM(
                input_size=input_size,
               lstm_hidden_size=params['lstm_hidden_size'],
                num_layers=params.get('num_layers', 1),
                num_classes=num_classes
            )

        elif model_name == 'MHAttention':
            model = MHAttention(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_heads=params['num_heads'],
                num_classes=num_classes
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        prev_loss = 0

        for epoch in tqdm(range(num_epochs)):
            total_loss = 0

            for x, y, x_lengths in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(x, x_lengths)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if abs(prev_loss - avg_loss) < 1e-4:
                break
            prev_loss = avg_loss

            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}", end="\r")

        return model

    # Mixed Models
    elif model_type == 'mixed':
        model_name = params['model_name']

        if model_name == 'BiLSTM_GAT_FC':
            model = BiLSTM_GAT_FC(
                input_size=params['input_size'],
                lstm_hidden_size=params['lstm_hidden_size'],
                lstm_layers=params['lstm_layers'],
                gat_hidden_size=params['gat_hidden_size'],
                num_classes=params['num_classes'],
                gat_heads=params['gat_heads'],
                dropout=params['dropout']
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        data = data.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        prev_loss = 0
        num_epochs = params['n_epochs']

        for epoch in tqdm(range(num_epochs)):
            model.train()
            optimizer.zero_grad()

            out = model(data)

            loss = criterion(
                out[data.train_mask],
                data.y[data.train_mask]
            )

            if abs(prev_loss - loss.item()) < 1e-4:
                break

            prev_loss = loss.item()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

        return model

    # Graph Models
    elif model_type == 'graph':
        model_name = params['model_name']
        num_categories = 12
        embed_dim = params['embed_dim']
        dropout = params['dropout']
        lstm_hidden_size = params['lstm_hidden_size']
        
        out_dim = torch.max(data.y).item() + 1

        if model_name == 'GCN':
            model = PopulationGCN(
                num_categories=num_categories,
                embed_dim=embed_dim,
                lstm_hidden_size=lstm_hidden_size,
                gcn_hidden_size=params['gcn_hidden_size'],
                out_dim=out_dim,
                dropout=dropout
            )

        elif model_name == 'GAT':
            model = PopulationGAT(
                num_categories=num_categories,
                embed_dim=embed_dim,
                lstm_hidden_size=lstm_hidden_size,
                gat_hidden_size=params['gat_hidden_size'],
                out_dim=out_dim,
                gat_heads=params['gat_heads'],
                dropout=dropout
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        data = data.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        prev_loss = 0
        num_epochs = params['n_epochs']

        for epoch in tqdm(range(num_epochs)):
            model.train()
            optimizer.zero_grad()

            out = model(data)

            loss = criterion(
                out[data.train_mask],
                data.y[data.train_mask]
            )

            if abs(prev_loss - loss.item()) < 1e-4:
                break

            prev_loss = loss.item()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

        return model
    


        
    
def test(model_type,model,data):
    if model_type == 'linear':
        X_train,X_test,y_train,y_test = data
        model_type = type(model).__name__
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_micro = f1_score(y_test, preds, average='micro')
        f1_macro = f1_score(y_test, preds, average='macro')
        f1_weighted = f1_score(y_test, preds, average='weighted')
        scores = {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': acc
        }
        return scores
    
    elif model_type == 'graph':
        model.eval()
        with torch.no_grad():
            logits = model(data)             
            preds = logits.argmax(dim=1)       

            y_test = data.y[data.test_mask].cpu().numpy()
            preds = preds[data.test_mask].cpu().numpy()

            acc = accuracy_score(y_test, preds)
            f1_micro = f1_score(y_test, preds, average='micro')
            f1_macro = f1_score(y_test, preds, average='macro')
            f1_weighted = f1_score(y_test, preds, average='weighted')
            scores = {
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'accuracy': acc
            }
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
    
    elif model_type == 'mixed':
        model.eval()
        with torch.no_grad():
            y_logits = model(data)
            y_pred = torch.argmax(y_logits, dim=1)
            
            test_mask = data.test_mask
            y_preds = y_pred[test_mask].cpu().numpy()
            ys = data.y[test_mask].cpu().numpy()
            
            scores = {
                'f1_micro': f1_score(ys, y_preds, average='micro'),
                'f1_macro': f1_score(ys, y_preds, average='macro'),
                'f1_weighted': f1_score(ys, y_preds, average='weighted'),
                'accuracy': accuracy_score(ys, y_preds)
            }
            
        return scores
  

