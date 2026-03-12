import torch

from First_model.linear_regression_model import LinearRegressionModel


def load_model():
    loaded_model_0 = LinearRegressionModel()
    loaded_model_0.load_state_dict(torch.load(f="models/model_0.pth"))
    return loaded_model_0
