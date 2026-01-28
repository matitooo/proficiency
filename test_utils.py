from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

def evaluate_linear(model,X,y,train_mask,test_mask):
    out = model.predict(X)
    train_acc = accuracy_score(out[train_mask],y[train_mask])
    train_f1 = f1_score(out[train_mask],y[train_mask],average='macro')
    test_acc = accuracy_score(out[test_mask],y[test_mask])
    test_f1 = f1_score(out[test_mask],y[test_mask],average='macro')
    print(f"Train Accuracy :{train_acc:.4f} Train F1 :{train_f1:.4f}")
    print(f"Test Accuracy :{test_acc:.4f} Test F1 :{test_f1:.4f}")
    return train_acc,train_f1,test_acc,test_f1


def evaluate_gcn(model, data):
    model.eval()
    out = model.predict(data)  # shape: (N,)

    y = data.y
    train_mask = data.train_mask
    test_mask  = data.test_mask

    y = y.cpu().numpy()
    train_mask = train_mask.cpu().numpy()
    test_mask = test_mask.cpu().numpy()
    out = out.cpu().numpy()

    train_acc = accuracy_score(out[train_mask], y[train_mask])
    train_f1  = f1_score(out[train_mask], y[train_mask], average='macro')

    test_acc = accuracy_score(out[test_mask], y[test_mask])
    test_f1  = f1_score(out[test_mask], y[test_mask], average='macro')

    print(f"Train Accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Test  Accuracy: {test_acc:.4f} | Test  F1: {test_f1:.4f}")

    return train_acc, train_f1, test_acc, test_f1

