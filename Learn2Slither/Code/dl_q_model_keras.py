import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DLQModel(nn.Module):
    def __init__(self, state_shape, nr_actions):
        super(DLQModel, self).__init__()
        self.model = nn.Sequential(
                      nn.Linear(state_shape, 128),  # Input layer
                      nn.ReLU(), # Add activation
                      nn.Linear(128, 64),  # Hidden layer
                      nn.ReLU(), # Add activation
                      nn.Linear(64, nr_actions)  # Output layer
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a numpy array or a PyTorch tensor")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim > 2:
            raise ValueError("Input tensor must be 1D or 2D")
        x = x.view(x.size(0), -1) # Flatten the input
        return self.model(x)  # Set the model to training mode
        
    
    def fit(self, X, Y, epochs=1000, batch_size=32, learning_rate=0.001):
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            optimizer.zero_grad()  # Zero the gradients

            outputs = self.forward(X_tensor)  # Forward pass
            loss = criterion(outputs, Y_tensor)  # Compute loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
