from sklearn.metrics import roc_auc_score, classification_report

def evaluate(model, X, y):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    print("ROC-AUC:", roc_auc_score(y, proba))
    print(classification_report(y, preds))
