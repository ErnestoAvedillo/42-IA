import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_LAYERS = 2
NUMBER_OF_NEURONS = 64


class DLQModel(nn.Module):
    def __init__(self, state_shape, nr_actions, gpu_device=0):
        super(DLQModel, self).__init__()
        layers = []
        layers.append(nn.Linear(state_shape, NUMBER_OF_NEURONS))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(NUMBER_OF_NEURONS, NUMBER_OF_NEURONS))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(NUMBER_OF_NEURONS, nr_actions))

        if torch.cuda.is_available():
            assert gpu_device < torch.cuda.device_count(), (
                f"GPU device {gpu_device} is not available."
            )
        self.device = torch.device(f'cuda:{gpu_device}'
                                   if torch.cuda.is_available() else 'cpu')
        self.model = nn.Sequential(*layers).to(self.device)

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X
        X_tensor = X_tensor.to(self.device)
        if X_tensor.ndim == 1:
            X_tensor = X_tensor.unsqueeze(0)
        elif X_tensor.ndim > 2:
            raise ValueError("Input tensor must be 1D or 2D")
        X_tensor = X_tensor.view(X_tensor.size(0), -1)  # Flatten the input
        output = self.model(X_tensor)  # Set the model to training mode
        return output

    def fit(self, X, Y, epochs=1000, batch_size=32, learning_rate=0.001):
        torch.autograd.set_detect_anomaly(True)
        # Convert numpy arrays to PyTorch tensors
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X
        if not isinstance(Y, torch.Tensor):
            Y_tensor = torch.tensor(Y, dtype=torch.float32)
        else:
            Y_tensor = Y
        X_tensor = X_tensor.to(self.device)
        Y_tensor = Y_tensor.to(self.device)
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            optimizer.zero_grad()  # Zero the gradients

            outputs = self.forward(X_tensor)  # Forward pass
            loss = criterion(outputs, Y_tensor)  # Compute loss

            loss.backward(retain_graph=True)  # Backward pass
            optimizer.step()  # Update weights

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
