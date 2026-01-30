import joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from .config import TARGET, TEST_SIZE, MODEL_PATH
from .data_loader import load_data
from .preprocessing import clean_data
from .features import build_features


def train():

    df = load_data()
    df = clean_data(df)
    df = build_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    model = CatBoostClassifier(verbose=0)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    return model, X_test, y_test
