import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from policy_network import PolicyNetwork

import itertools
    

# Generate all possible board configurations
def generate_board_configurations():
    empty_board = [0] * 9
    return list(itertools.product([-1, 0, 1], repeat=9))

def generate_board_configurations_with_a_win():
    all_boards = generate_board_configurations()
    new_boards = []
    for board in all_boards:
        if(determine_winning_moves(board)!=-1):
            new_boards.append(board)
    return new_boards

# Define winning combinations (indices)
win_combinations = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6]              # Diagonals
]

# Check if a player has won
def check_winner(player):
    for combo in win_combinations:
        if all(board[i] == player for i in combo):
            return True
    return False

# Given the board (9 element array/tensor), returns a list of the winning move indices, or -1 is there are none
def determine_winning_moves(board):
    board = list(board)
    # Check winning moves for 1s (our moves)
    winning_moves_list = []
    for i, val in enumerate(board):
        if val == 0:
            board[i] = 1
            if check_winner(1):
                winning_moves_list.append(i)
            board[i] = 0
    
    if(len(winning_moves_list)==0):
        return -1
    
    return winning_moves_list

# Generate dataset
board_configurations = generate_board_configurations()
# board_configurations = generate_board_configurations_with_a_win()
dataset = []

# For every possible board, 
for board in board_configurations:
    # Find if there is winning moves for this board
    label = determine_winning_moves(board) # Either list of moves or -1 
    if label != -1:
        # Append a tuple of the board state and the winning moves
        dataset.append((board, label))

# We now have a dataset with every possible board state and the desired indices of the board state
    # for which the neural network should rate the highest

# We need to give this data to the neural network, and let it make a prediction, 
    # then reward it if it chooses a winning move (if there is a winning move)

# Instantiate the neural network
net = PolicyNetwork()
net.load_state_dict(torch.load('policy_network_model.pth'))

# Convert data to tensors
X_train = torch.tensor([board for board, _ in dataset], dtype=torch.float32)
y_train = []

for _, label in dataset:
    if isinstance(label, list):  # If label is a list of winning moves
        # Create a list where each element corresponds to the index of the winning move in the board state
        label_list = [-1] * 9  # Initialize with -1 (no winning move)
        for move in label:
            label_list[move] = 1  # Set the index of the winning move to 1
        y_train.append(label_list)
    else:  # If label is -1 (no winning moves)
        y_train.append([-1] * 9)  # Add a list of -1s

# Convert y_train to tensor
y_train = torch.tensor(y_train, dtype=torch.float32) 

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Define optimizer
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Create DataLoader
batch_size = 32
dataset = TensorDataset(X_train, y_train)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop to train the network
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for data, labels in data_loader:
        optimizer.zero_grad()
        outputs = net(data)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(data_loader)}")


# Save trained model
torch.save(net.state_dict(), 'policy_network_model.pth')

