import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import f1_score

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.pipeline import make_pipeline
from src.config import MODEL_PATH, TARGET, RANDOM_STATE, TEST_SIZE, THRESHOLD

def train_and_save():
    df = load_data()
    df = clean_data(df)

    # y
    y = df[TARGET].astype(int)

    # X raw (важно: оставляем сырые категориальные!)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipe = make_pipeline(X_train)

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds))

    # Диагностика порога (боевой THRESHOLD не меняем)
    grid = [i / 100 for i in range(1, 100)]
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_test, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"Best F1 threshold on holdout: {best_t:.2f} (F1={best_f1:.4f}); current THRESHOLD={THRESHOLD}")

    # ensure output dir exists
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
