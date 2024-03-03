import torch
from neuralNetwork.policy_network import PolicyNetwork

# Load the saved model
model = PolicyNetwork()
model.load_state_dict(torch.load('policy_network_model.pth'))
model.eval()  # Set the model to evaluation mode

# Assuming you have an input_board as a torch tensor
input_board = torch.tensor([[1, 1, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)

# Make predictions using the loaded model
predicted_moves = None
with torch.no_grad():
    predicted_probabilities = model(input_board)
    predicted_move = torch.argmax(predicted_probabilities, dim=1)

print("Predicted Probabilities:", predicted_probabilities)
print("Predicted Moves:", predicted_move)
