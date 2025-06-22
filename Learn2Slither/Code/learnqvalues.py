import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("firstqvalues.csv")
x = df.iloc[:, :-4].values
y = df.iloc[:, -4:].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = torch.nn.Sequential(
    torch.nn.Linear(xtrain.shape[1], 128),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(128),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(64),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, ytrain.shape[1])
)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  

for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    
    inputs = torch.tensor(xtrain, dtype=torch.float32)
    targets = torch.tensor(ytrain, dtype=torch.float32)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    test_inputs = torch.tensor(xtest, dtype=torch.float32)
    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, torch.tensor(ytest, dtype=torch.float32))
    print(f'Test Loss: {test_loss.item()}')
    for i in range(len(xtest)):
        print("Test Sample", i)
        print(f"Input: {xtest[i]}")
        print(f"Predicted Q-values: {test_outputs[i].numpy()}.")
        print(f"Actual    Q-values: {ytest[i]}")
