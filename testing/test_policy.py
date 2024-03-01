import torch
import numpy as np
from policy_network import PolicyNetwork

# Assuming you have already instantiated your policy network
policy_network = PolicyNetwork(input_size=83, hidden_size=30, output_size=1)  # Adjusted input_size to 83

# Assuming 'board_state' is your 9x9 numpy array representing the board state
# and 'move' is the tuple coordinate representing the move
board_state = np.zeros((9, 9))
move = (0,0)

# Flatten the board state
flattened_board_state = board_state.flatten()

# Concatenate the flattened board state with the move tuple
input_features = np.concatenate((flattened_board_state, move), axis=None)

# Convert the input features to a PyTorch tensor
input_tensor = torch.tensor(input_features, dtype=torch.float32)
input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

# Forward pass through the policy network
output_value = policy_network(input_tensor)

# Extract the output value
output_value = output_value.item()

print(output_value)
