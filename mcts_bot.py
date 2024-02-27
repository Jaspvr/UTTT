import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy_network import PolicyNetwork


class TreeNode:
  def __init__(self,
               state,
               parent=None,
               move=None,
               active_box=None,
               valid_moves=None,
               visits=None,
               outcome=None,
               depth=None):
    self.state = state
    self.parent = parent
    self.init_move = move
    self.children = [] 
    self.visits = 0  #initial is always 0
    self.value = 0  # initial is always 0
    self.active_box = active_box
    self.valid_moves = valid_moves
    self.depth = depth  # Root is at depth 0
    self.outcome = outcome

class Jaspers_MCTS_Agent:
  ''' Monte Carlo Search Tree UTTT player, move function returns it's next move '''
  def __init__(self, name: str = 'mcts_bot', debug=True):
    self.name = name
    self.policy_network = PolicyNetwork(input_size=2, hidden_size=30, output_size=80)  # Initialize the policy network

  def move(self, board_dict: dict) -> tuple:
    ''' Return the move that the agent wants to make given the current board state and available moves '''
    board_state = board_dict['board_state']  # 9x9 numpy array
    active_box = board_dict['active_box']  # (x, y) tuple of coordinates of the smaller 3x3 game
    valid_moves = board_dict['valid_moves']  # List of all valid moves, i.e list of tuples - BIG BOARD
    
    root_state = board_state
    root_node = TreeNode(root_state,
                        active_box=active_box,
                        valid_moves=valid_moves,
                        visits=1,
                        depth=0, 
                        )

    # Search tree loop: builds out a game tree by performing a leaf node selection, expansion, simulation, and 
    #   back propogation starting at the root node
    count = 0
    while count < 10:

      #Selection phase: Traverse from the root node to a leaf node
      selected_leaf_node = self.selection(root_node)  # We now have the leaf node to work with

      # Expansion Phase: our selected node is a leaf node, we have to create a children with all the possible valid moves from that place
      if(selected_leaf_node.visits != 0): #Account for case of not visited the node before (add 1 since this is a visit)
        self.expansion(selected_leaf_node, selected_leaf_node.valid_moves)  #We have now expanded. We should simulate
      else:
        selected_leaf_node.visits += 1

      # Simulation Phase: simulate down the tree until terminal state is reached
      reward = self.simulation(selected_leaf_node)  #return the value of the game end (win, draw, loss)

      # Backpropogate: Propogate game outcome from simulation up the tree for node UCB scores
      self.backpropogate(selected_leaf_node, reward)
      count += 1
    
    # Find the best move (currently accessing the highest number of visits)
    max_value = -1
    max_child = None
    for child in root_node.children:
      if child.visits > max_value:
        max_child = child
  
    move_to_make = max_child.init_move  # Get the move cooresponding to the child node

    return move_to_make

  def selection(self, node):  
        ''' Select the next node to explore using UCB and policy network '''
        exploration_constant = 1.55

        while not all(child is None for child in node.children):
            ucb_values = [
                self.calculate_ucb(child, exploration_constant, node.visits)
                for child in node.children
            ]

            # Use the policy network to estimate the value of each child node
            # Only use the policy network for when the mini game is defined
            if len(node.children) <= 9:
              # Calculate the row and column indices of the sub-box
              m = node.children[0].init_move
              sub_row = m[0] // 3
              sub_col = m[1] // 3
              minibox = (sub_row, sub_col)
              mini_board = self.pull_mini_board(node.state, minibox)
              print(mini_board)
              policy_values = []
              for child in node.children:
                  mini_move = self.map_to_mini_box(m)
                  policy_value = self.policy_network.forward(child.init_move, mini_move, mini_board)
                  policy_values.append(policy_value)

            # Combine UCB values and policy values to select the child node
            combined_values = [ucb + policy for ucb, policy in zip(ucb_values, policy_values)]
            selected_index = combined_values.index(max(combined_values))
            node = node.children[selected_index]

        return node

  def calculate_ucb(self, node, exploration_constant, parent_visits):
    ''' Calculate the ucb score for selecting the best node during selection process '''
    if node.visits == 0:
      return float('inf')  # Prioritize unvisited nodes
    else:
      exploitation_term = node.value / node.visits
      exploration_term = exploration_constant * math.sqrt(
          math.log(parent_visits) / node.visits)
      return exploitation_term + exploration_term 

  def backpropogate(self, selected_leaf_node, reward):
    ''' Return add the value of the result of the simulation up the tree '''
    while selected_leaf_node.parent is not None:
      selected_leaf_node.value += reward
      selected_leaf_node.visits += 1
      selected_leaf_node = selected_leaf_node.parent

  def simulation(self, selected_leaf_node):
    ''' Simulate the game from the selected leaf node '''
    # Get whos turn, board state, active box, valid moves
    whos_move_value = self.whos_move_value(selected_leaf_node)
    valid_moves = selected_leaf_node.valid_moves
    board_state = selected_leaf_node.state
    active_box = selected_leaf_node.active_box

    # Simulate the game until it reaches a terminal state
    count = 0
    while True:  
      if self.get_outcome(board_state) != -3: # -3 means that the game is not over
        break
      move = random.choice(valid_moves)
    
      # Update board state, valid moves, active box with make_move
      board_state, valid_moves, active_box = self.make_move(
          move, board_state, whos_move_value)

      # Update whos turn
      whos_move_value *= -1
      count+=1

    # Now we have reached a terminal state, set the value of the game to the leaf node
    outcome_value = self.get_outcome(board_state)
    return outcome_value


  def active_box_after_move(self, move, board_state):
    ''' Returns the active box after the move is made '''

    # Each tuple in a given list is a move that will lead to the next active box being the list name ((6, 6) maps to big board box (2, 2))
    box22 = [(6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7),(8, 8)]
    box21 = [(6, 3), (6, 4), (6, 5), (7, 3), (7, 4), (7, 5), (8, 3), (8, 4),(8, 5)]
    box20 = [(6, 0), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2), (8, 0), (8, 1),(8, 2)]
    box12 = [(3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7),(5, 8)]
    box11 = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4),(5, 5)]
    box10 = [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1),(5, 2)]
    box02 = [(0, 6), (0, 7), (0, 8), (1, 6), (1, 7), (1, 8), (2, 6), (2, 7),(2, 8)]
    box01 = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),(2, 5)]
    box00 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1),(2, 2)]

    if move in box00:
      new_active_box = (0, 0)
    elif move in box01:
      new_active_box = (0, 1)
    elif move in box02:
      new_active_box = (0, 2)
    elif move in box10:
      new_active_box = (1, 0)
    elif move in box11:
      new_active_box = (1, 1)
    elif move in box12:
      new_active_box = (1, 2)
    elif move in box20:
      new_active_box = (2, 0)
    elif move in box21:
      new_active_box = (2, 1)
    elif move in box22:
      new_active_box = (2, 2)

    return new_active_box

  def get_outcome(self, board_state):
    ''' Big game outcome '''
    #Check for which mini games are over and their result
    #Go through mini games 00 to 22, check if they are over,
    mini_game_tuples = [(i, j) for i in range(3)
                        for j in range(3)]  #List of mini board coordinates
    mini_game_outcomes = []
    for mini_game_tuple in mini_game_tuples:
      mini_game_state = self.pull_mini_board(board_state, mini_game_tuple)
      mini_game_outcome = self.subgame_terminated(
          mini_game_state)  #Either 1, 0.5 ,or 0, or -3
      if mini_game_outcome == -3:
        mini_game_outcomes.append(0)
      elif mini_game_outcome == 1:  #win
        mini_game_outcomes.append(1)
      elif mini_game_outcome == 0.5:  #draw
        mini_game_outcomes.append(0.5)
      elif mini_game_outcome == 0:  #loss
        mini_game_outcomes.append(-1)

    #Make the mini game outcomes into a 3x3 np.array instead of a list
    # Put the result in a new 3x3 representing the big board
    mini_game_outcomes_matrix = np.array(mini_game_outcomes).reshape(3, 3)
    #Check big game
    big_game_outcome = self.big3x3_terminated(mini_game_outcomes_matrix)
    return big_game_outcome
  
  def big3x3_terminated(self, mini_board):
    ''' Check if 3x3 representing the big board game is over '''
    # Check rows and columns
    for i in range(3):
      if np.all(mini_board[i, :] == 1) or np.all(mini_board[:, i] == 1):
        return 1  # Player 1 wins
      elif np.all(mini_board[i, :] == -1) or np.all(mini_board[:, i] == -1):
        return 0  # Player 2 wins

    # Check diagonals
    if np.all(np.diag(mini_board) == 1) or np.all(
        np.diag(np.fliplr(mini_board)) == 1):
      return 1  # Player 1 wins
    elif np.all(np.diag(mini_board) == -1) or np.all(
        np.diag(np.fliplr(mini_board)) == -1):
      return 0  # Player 2 wins

    # Check for a draw
    if np.count_nonzero(mini_board == 0) == 0:
      return 0.5  # Draw

    # If no winner yet
    return -3

  def whos_move_value(self, node):

    whos_move_value = node.depth % 2  #0 is us(move is 1 on board), 1 is opponent (then move is -1 on the board)
    if whos_move_value == 0:
      whos_move_value = 1
    elif whos_move_value == 1:
      whos_move_value = -1

    return whos_move_value

  #approved
  # def selection(self, node):  #QJKAPROVE #Approve
  #   ''' Look at children, go to one with highest ucb, go until reach leaf node, this is the selected node for expansion '''
  #   # Define the exploration constant
  #   exploration_constant = 1.55

  #   # Traverse down the tree until a leaf node is reached
  #   while not all(child is None for child in
  #                 node.children):  #at a leaf node when all children are None
  #     # Calculate UCB values for each child node
  #     ucb_values = [
  #         self.calculate_ucb(child, exploration_constant, node.visits)
  #         for child in node.children
  #     ]  #cheeky list comprehension

  #     # Select the child node with the highest UCB value
  #     selected_index = ucb_values.index(max(ucb_values))
  #     node = node.children[selected_index]

  #   return node

  # def selection(self, node):
  #       '''Select the best child node based on UCB and policy network'''
  #       while not all(child is None for child in node.children):
  #           ucb_values = [
  #               self.calculate_ucb(child, node.visits)
  #               for child in node.children
  #           ]

  #           # Obtain action probabilities from the policy network
  #           state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
  #           action_probs = self.policy_network(state_tensor)
  #           action_probs = action_probs.squeeze().detach().numpy()

  #           # Combine UCB values with action probabilities
  #           combined_values = [ucb + action_probs[i] for i, ucb in enumerate(ucb_values)]

  #           # Select the child node with the highest combined value
  #           selected_index = combined_values.index(max(combined_values))
  #           node = node.children[selected_index]

  #       return node

  def expansion(self, leaf_node, valid_moves):
    """ Expand the tree by creating child nodes for the selected leaf node.Assign the leaf node as the parent of each child node."""
    new_depth = leaf_node.depth + 1
    for move in valid_moves:  # For [new node that we need to create for the move] in [valid moves]
      # Create a new state based on the move
      whos_move_value = self.whos_move_value(leaf_node)
      new_state, new_valid_moves, new_active_box = self.make_move(
          move, leaf_node.state, whos_move_value)

      # Create a new child node with the updated state and link it to the leaf node
      new_node = TreeNode(state=new_state,
                          parent=leaf_node,
                          move=move,
                          active_box=new_active_box,
                          valid_moves=new_valid_moves,
                          depth=new_depth)

      # Add the new child node to the children list of the leaf node
      leaf_node.children.append(new_node)

  def make_move(self, move, current_state, whos_move_value): #tested
    ''' current state is a 9x9, move is a tuple, whos_move_value is either 1 or -1 '''
    ''' returns new state after move is made '''
    # Get the new state after the move
    move_x = move[0]
    move_y = move[1]
    new_state = current_state
    new_state = np.copy(current_state)
    new_state[move_x, move_y] = whos_move_value #set the square to value of current player

    # Get the active box
    # This still works even with the rotation
    # MAPS TO THIS BOX
    box00 = (0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3), (6, 6)
    box01 = (0, 1), (0, 4), (0, 7), (3, 1), (3, 4), (3, 7), (6, 1), (6, 4), (6, 7)
    box02 = (0, 2), (0, 5), (0, 8), (3, 2), (3, 5), (3, 8), (6, 2), (6, 5), (6, 8)
    box10 = (1, 0), (1, 3), (1, 6), (4, 0), (4, 3), (4, 6), (7, 0), (7, 3), (7, 6)
    box11 = (1, 1), (1, 4), (1, 7), (4, 1), (4, 4), (4, 7), (7, 1), (7, 4), (7, 7)
    box12 = (1, 2), (1, 5), (1, 8), (4, 2), (4, 5), (4, 8), (7, 2), (7, 5), (7, 8)
    box20 = (2, 0), (2, 3), (2, 6), (5, 0), (5, 3), (5, 6), (8, 0), (8, 3), (8, 6)
    box21 = (2, 1), (2, 4), (2, 7), (5, 1), (5, 4), (5, 7), (8, 1), (8, 4), (8, 7)
    box22 = (2, 2), (2, 5), (2, 8), (5, 2), (5, 5), (5, 8), (8, 2), (8, 5), (8, 8)

    new_active_box = None
    if move in box00:
      new_active_box = (0, 0)
    elif move in box01:
      new_active_box = (0, 1)
    elif move in box02:
      new_active_box = (0, 2)
    elif move in box10:
      new_active_box = (1, 0)
    elif move in box11:
      new_active_box = (1, 1)
    elif move in box12:
      new_active_box = (1, 2)
    elif move in box20:
      new_active_box = (2, 0)
    elif move in box21:
      new_active_box = (2, 1)
    elif move in box22:
      new_active_box = (2, 2)

    # Now we have the active box for next move

    # Get all of the valid moves
    all_moves = []
    new_mini_board = self.pull_mini_board(new_state, new_active_box)

    if self.subgame_terminated(new_mini_board) != -3:
      new_active_box = (-1, -1)

    # Case: the subgame is terminated - we have go to any available space on the board
    if new_active_box == (-1, -1):
      tuple_list = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
      tuples_revised = []
      for tuple1 in tuple_list:
        new_mini_board = self.pull_mini_board(new_state, tuple1)
        terminated_val = self.subgame_terminated(new_mini_board)
        if terminated_val == -3:
          tuples_revised.append(tuple1)

      # We now have a revised list of tuples of non finished games
      for tuple_revised in tuples_revised:
        # We want to get all tuple places on in any of these squares
        all_moves.append(self.get_coordinates_in_submatrix(tuple_revised))

      # Check for spaces of '0'. these are our valid moves
      new_valid_moves = []
      for move_list in all_moves:
        for move_tuple in move_list:
          if new_state[move_tuple[0], move_tuple[1]] == 0:
            new_valid_moves.append(move_tuple)

      return [new_state, new_valid_moves, new_active_box]

    else:
        #Get valid moves in new active box
        new_mini_board = self.pull_mini_board(new_state, new_active_box)
        new_valid_moves = self.from_mini_to_big(new_mini_board,new_active_box)  # new_valid_moves is in terms of the 9x9 matrix

        #Have a list of valid moves in the new mini board
        return [new_state, new_valid_moves, new_active_box] 
    
  def get_coordinates_in_submatrix(self, coord_tuple):
    ''' Return all coordinates of a subgame given that subgame's location on the big game '''
    submatrix_coordinates = []
    for row in range(3):
        for col in range(3):
            submatrix_coordinates.append((coord_tuple[0]*3 + row, coord_tuple[1]*3 + col))
    return submatrix_coordinates

  def from_mini_to_big(self, new_mini_board, new_active_box):
    ''' Mini game moves to big game moves '''
    box_mapping = {
        (0, 0): (0, 0),
        (0, 1): (0, 3),
        (0, 2): (0, 6),
        (1, 0): (3, 0),
        (1, 1): (3, 3),
        (1, 2): (3, 6),
        (2, 0): (6, 0),
        (2, 1): (6, 3),
        (2, 2): (6, 6)
    }

    new_valid_moves = []

    for i, row in enumerate(new_mini_board):
      for j, element in enumerate(row):
        if element == 0:
          new_valid_moves.append((i, j))

    valid_moves_9x9 = []
    for subbox in new_valid_moves:
      mapped_row, mapped_col = box_mapping[new_active_box]
      valid_moves_9x9.append((mapped_row + subbox[0], mapped_col + subbox[1]))

    return valid_moves_9x9

  def pull_mini_board(self, board_state: np.array, mini_board_index: tuple) -> np.array:
    ''' extracts a mini board from the 9x9 given the its index'''
    temp = board_state[mini_board_index[0] * 3:(mini_board_index[0] + 1) * 3,
                       mini_board_index[1] * 3:(mini_board_index[1] + 1) * 3]
    return temp

  def subgame_terminated(self, mini_board):
    ''' Check if small game is over '''
    # Check rows and columns
    for i in range(3):
      if np.all(mini_board[i, :] == 1) or np.all(mini_board[:, i] == 1):
        return 1  # Player 1 wins
      elif np.all(mini_board[i, :] == -1) or np.all(mini_board[:, i] == -1):
        return 0  # Player 2 wins

    # Check diagonals
    if np.all(np.diag(mini_board) == 1) or np.all(
        np.diag(np.fliplr(mini_board)) == 1):
      return 1  # Player 1 wins
    elif np.all(np.diag(mini_board) == -1) or np.all(
        np.diag(np.fliplr(mini_board)) == -1):
      return 0  # Player 2 wins

    # Check for a draw
    if np.count_nonzero(mini_board == 0) == 0:
      return 0.5  # Draw

    # If no winner yet
    return -3

  def map_to_mini_box(self, move):
    # Get the row and column indices of the move within the mini-box
    mini_row = move[0] % 3
    mini_col = move[1] % 3
    return (mini_row, mini_col)

  def print_tree(self, node, level=0):
    """ Prints the tree from the given node. """
    if node is not None:
      print(
          "  " * level +
          f"Move: {node.init_move}, Value: {node.value}, Visits: {node.visits}"
      )
      for child in node.children:
        self.print_tree(child, level + 1)


## For Testing/debugging

# Mock input data
# board_dict = {
#     'board_state': np.zeros((9, 9)),  # Example of a 9x9 board with all zeros
#     'active_box': (1, 1),  # Example of the active box
#     'valid_moves': [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4),
#                     (5, 5)]  # Example of valid moves
# }
# # First opponent move
# board_dict['board_state'][4, 4] = -1
# board_state = board_dict['board_state']

# # Instantiate MCTS agent
# mcts_agent = Jaspers_MCTS_Agent()

# # Test expansion - Print the tree
# # root_node = TreeNode(board_dict['board_state'])
# # mcts_agent.expansion(root_node, [(1, 1), (2, 2), (3, 3)])  # Example valid moves
# # mcts_agent.print_tree(root_node)

# # Call move function
# selected_move = mcts_agent.move(board_dict)

# # Inspect output
# print("Selected move:", selected_move)