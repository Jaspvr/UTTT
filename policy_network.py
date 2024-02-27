import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, mini_move, mini_board):
        ''' x is the actual move on the big board'''
        print(x, mini_move, mini_board)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def map_to_mini_box(move):
        # Get the row and column indices of the move within the mini-box
        mini_row = move[0] % 3
        mini_col = move[1] % 3
        return (mini_row, mini_col)
# class MCTS:
#     def __init__(self, policy_network, exploration_constant):
#         self.policy_network = policy_network
#         self.exploration_constant = exploration_constant

    # def selection(self, node):
    #     '''Select the best child node based on UCB and policy network'''
    #     while not all(child is None for child in node.children):
    #         ucb_values = [
    #             self.calculate_ucb(child, node.visits)
    #             for child in node.children
    #         ]

    #         # Obtain action probabilities from the policy network
    #         state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
    #         action_probs = self.policy_network(state_tensor)
    #         action_probs = action_probs.squeeze().detach().numpy()

    #         # Combine UCB values with action probabilities
    #         combined_values = [ucb + action_probs[i] for i, ucb in enumerate(ucb_values)]

    #         # Select the child node with the highest combined value
    #         selected_index = combined_values.index(max(combined_values))
    #         node = node.children[selected_index]

    #     return node

    # def calculate_ucb(self, node, parent_visits):
    #     '''Calculate the UCB score for selecting the best node during the selection process'''
    #     if node.visits == 0:
    #         return float('inf')  # Prioritize unvisited nodes
    #     else:
    #         exploitation_term = node.value / node.visits
    #         exploration_term = self.exploration_constant * math.sqrt(
    #             math.log(parent_visits) / node.visits)
    #         return exploitation_term + exploration_term

# # Example usage:
# input_size = 100  # Example input size
# hidden_size = 64  # Example hidden size
# output_size = 10  # Example output size
# exploration_constant = 1.55  # Example exploration constant

# # Initialize policy network
# policy_network = PolicyNetwork(input_size, hidden_size, output_size)

# # Initialize MCTS with policy network and exploration constant
# mcts = MCTS(policy_network, exploration_constant)

# # Perform selection using MCTS with policy network
# selected_node = mcts.selection(node)
