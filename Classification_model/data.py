# import pandas as pd
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)


def random_split():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})  # type: ignore
