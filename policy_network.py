import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        ''' x is the tensor for the neural network to make predictions on'''
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    

# Plan: Pass in the active mini board, output a rating for each square
    # find which ones coorespond to valid moves, and then add the rating to the ucb for each valid moves
    # train with games that are one move away from winning
    
train_input = 


# Assume you have your training data (train_input, train_target) and validation data (val_input, val_target) ready

# Convert your data into PyTorch tensors
train_input_tensor = torch.tensor(train_input, dtype=torch.float32)
train_target_tensor = torch.tensor(train_target, dtype=torch.long)
val_input_tensor = torch.tensor(val_input, dtype=torch.float32)
val_target_tensor = torch.tensor(val_target, dtype=torch.long)

# Create a DataLoader for training and validation data
train_dataset = TensorDataset(train_input_tensor, train_target_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(val_input_tensor, val_target_tensor)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define your model
input_size = # Define your input size
hidden_size = # Define your hidden size
output_size = # Define your output size
model = PolicyNetwork(input_size, hidden_size, output_size)

# Choose your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss/len(val_loader)}, Val Acc: {100*correct/total}%')

# Test your model if needed

# Save your model if needed
torch.save(model.state_dict(), 'policy_network.pth')
