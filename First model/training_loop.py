import numpy as np
import torch
from linear_regression_model import LinearRegressionModel
from making_predictions import making_predictions
from plot_prediction import plot_loss_curves
from torch import nn
from train_split import train_split

torch.manual_seed(42)

# Model inicialization and data split
model_0 = LinearRegressionModel()
X_train, y_train, X_test, y_test = train_split()

# Loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 200  # Number of loops through the data

# Tracking different values
epoch_count = []
loss_values = []
test_loss_values = []

# 0. Loop through the data
for epoch in range(epochs):
    model_0.train()  # train mode in PyTorch sets all parameters that require gradients to require gradients

    # 1. forward pass on train data using the forward() method inside
    y_pred = model_0(X_train)

    # 2. Calculate the loss (how different are the model's predictions to the true values)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero the gradients of the optimizer (they accumulate by default)
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step()  # by default how the optimizer changes will accumulate through the loop so we have to zero them above in step 3 for the next iteration of the loop

    model_0.eval()  # turns off different settings in the model not needed for evaluation/testing (dropout/batchnNorm layers)
    with torch.inference_mode():  # turns off gradient tracking
        # 1. Do the forward pass
        test_pred = model_0(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())

making_predictions(model_0, X_train, y_train, X_test, y_test)

# np.array(torch.tensor(loss_values).numpy())
loss_values_np = torch.stack(loss_values).detach().numpy()
plot_loss_curves(epoch_count, loss_values_np, test_loss_values)
