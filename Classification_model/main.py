import torch
from CircleModel import CircleModel
from data import random_split

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_0 = CircleModel().to(device)
    X_train, X_test, y_train, y_test = random_split()

    with torch.inference_mode():
        untrained_preds = model_0(X_test.to(device))
