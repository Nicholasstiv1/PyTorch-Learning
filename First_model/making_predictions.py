import torch
from linear_regression_model import LinearRegressionModel
from plot_prediction import plot_predictions
from train_split import train_split


# Don't store the gradient data; faster than simply using model_0(X_train)
def making_predictions(model, X_train, y_train, X_test, y_test):
    with torch.inference_mode():
        y_preds = model(X_test)
    plot_predictions(X_train, y_train, X_test, y_test, y_preds)
