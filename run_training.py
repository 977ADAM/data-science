from src.train import train
from src.evaluate import evaluate

model, X_test, y_test = train()
evaluate(model, X_test, y_test)
