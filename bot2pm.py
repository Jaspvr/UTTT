import numpy as np
import math
import random  #ASK ABT THIS

#1: Ask and test make_move
#2: Simulate CHECK
#3: test simulate
#4: backpropogate
#5: test backpropogate
#6: loop over select, expand, sim, backprop
#7: test all together, fix bugs

#8: haha yay!!
#9: neural network?


class TreeNode:

  def __init__(self,
               state,
               parent=None,
               move=None,
               active_box=None,
               valid_moves=None,
               outcome=None,
               depth=None):  #pass in state, unsure about parent and move
    self.state = state
    self.parent = parent
    self.init_move = move
    self.children = []  #up to 81 (free board with everything open)
    self.visits = 0  #initial is always 0
    self.value = 0  # initial is always 0
    self.active_box = active_box
    self.valid_moves = valid_moves
    self.depth = depth  # Root is at depth 0
    self.outcome = outcome
    # self.outcome = state.get_outcome() #check and see if the game is over


class Jaspers_MCTS_Agent:

  def __init__(self, name: str = 'jaspers_bot', debug=True):
    self.name = name

  def move(self, board_dict: dict) -> tuple:
    board_state = board_dict['board_state']  #9x9 numpy array
    active_box = board_dict[
        'active_box']  #(x, y) tuple of coordinates of the smaller 3x3 game
    valid_moves = board_dict[
        'valid_moves']  #list of all valid moves, i.e list of tuples - BIG BOARD

    # count = 0
    # while count < 5:

    #Selection phase: Traverse from the root node to a leaf node
    root_state = board_state
    root_node = TreeNode(root_state,
                         active_box=active_box,
                         valid_moves=valid_moves,
                         depth=0)
    selected_leaf_node = self.selection(
        root_node
    )  # we now have the leaf node to work with. for the root node, selection function will just return back the same node

    # Expansion Phase: our selected node is a leaf node, we have to create its children with all the possible valid moves from that place
    self.expansion(selected_leaf_node, selected_leaf_node.valid_moves
                   )  #We have now expanded. We should simulate

    # self.print_tree(root_node) ##LOOKS GOOD FOR NOW UP TO HERE

    # Simulation Phase: simulate down the tree until terminal state is reached
    reward = self.simulation(
        selected_leaf_node
    )  #return the value of the game end (win, draw, loss)

    print(reward)

    # Backpropogate: Propogate ___ up the tree for node UCB scores (i think)
    # self.backpropogate(selected_leaf_node, reward)

    # print(reward)

    # count += 1

    return valid_moves[0]  # placeholder

  def backpropogate(self, selected_leaf_node, reward):
    #BACKPROPOGATE, adding outcome and visit count up
    while selected_leaf_node.parent is not None:
      #naming here is not consistent; selected_leaf_node, but after first while iteration, not a leaf node, since backpropogated.
      selected_leaf_node.parent.outcome += reward
      selected_leaf_node.parent.visits += 1
      selected_leaf_node = selected_leaf_node.parent

  def simulation(self, selected_leaf_node):
    ''' Simulate the game from the selected leaf node '''
    # get whos turn, board state, active box, valid moves
    whos_move_value = self.whos_move_value(selected_leaf_node)

    valid_moves = selected_leaf_node.valid_moves
    # print(valid_moves)
    board_state = selected_leaf_node.state
    active_box = selected_leaf_node.active_box

    # Simulate the game until it ends
    count = 0
    flag = False
    while self.get_outcome(board_state) == -3:  #-3 is game is not over
      if(len(valid_moves)==1 or flag):
        if(flag):
            flag = False
        return
      else:
        #randomly play one move
        move = random.choice(valid_moves)
        # if move leads to finished game, then skip it
        # What is a move? it is a tuple of coordinates. if this tuple leads to a finished game, then skip it
        move_leads_to_finished_game = self.move_leads_to_a_finished_game(move, board_state)
        if(move_leads_to_finished_game):
            if(count >4):
                flag = True
            count+=1
            continue #goes back and gets a new move
            
      #update whos turn
      whos_move_value *= -1
      #update board state, valid moves, active box with make_move
      board_state, valid_moves, active_box = self.make_move(
          move, board_state, whos_move_value)

    # Now we have reached a terminal state, set the value of the game to the leaf node
    outcome_value = self.get_outcome(board_state)
    return outcome_value

  def move_leads_to_a_finished_game(self, move, board_state):
    ''' Check if the move leads to a finished game '''
    new_active_box = self.active_box_after_move(move, board_state)
    #if that box is done. return True
    if self.subgame_terminated != -3: # game is done
      return True
    
    return False
    
  

  def active_box_after_move(self, move, board_state):
    ''' returns the active box after the move is made '''
    #GOES TO THIS BOX

    subbox = move
  
    box22 = [(6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7), (8, 8)]
    box21 = [(6, 3), (6, 4), (6, 5), (7, 3), (7, 4), (7, 5), (8, 3), (8, 4),(8, 5)]
    box20 = [(6, 0), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2), (8, 0), (8, 1),(8, 2)]
    box12 = [(3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7),(5, 8)]
    box11 = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4),(5, 5)]
    box10 = [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1),(5, 2)]
    box02 = [(0, 6), (0, 7), (0, 8), (1, 6), (1, 7), (1, 8), (2, 6), (2, 7),(2, 8)]
    box01 = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),(2, 5)]
    box00 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1),(2, 2)]

    if subbox in box00:
      new_active_box = (0, 0)
    elif subbox in box01:
      new_active_box = (0, 1)
    elif subbox in box02:
      new_active_box = (0, 2)
    elif subbox in box10:
      new_active_box = (1, 0)
    elif subbox in box11:
      new_active_box = (1, 1)
    elif subbox in box12:
      new_active_box = (1, 2)
    elif subbox in box20:
      new_active_box = (2, 0)
    elif subbox in box21:
      new_active_box = (2, 1)
    elif subbox in box22:
      new_active_box = (2, 2)
      
    return new_active_box

  def get_outcome(self, board_state):
    ''' Big game outcome '''
    #Check for which mini games are over and their result
    #Go through mini games 00 to 22, check if they are over,
    mini_game_tuples = [(i, j) for i in range(3) for j in range(3)]  #List of mini board coordinates
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
    big_game_outcome = self.subgame_terminated(mini_game_outcomes_matrix)
    return big_game_outcome

  def whos_move_value(self, node):
    whos_move_value = node.depth % 2  #0 is us(move is 1 on board), 1 is opponent (then move is -1 on the board)
    if whos_move_value == 0:
      whos_move_value = 1
    elif whos_move_value == 1:
      whos_move_value = -1

    return whos_move_value

  def selection(self, node):  #QJKAPROVE
    ''' Look at children, go to one with highest ucb, go until reach leaf node, this is the selected node for expansion '''
    # Define the exploration constant
    exploration_constant = 1.41

    # Traverse down the tree until a leaf node is reached
    while not all(child is None for child in
                  node.children):  #at a leaf node when all children are None
      # Calculate UCB values for each child node
      ucb_values = [
          self.calculate_ucb(child, exploration_constant, node.visits)
          for child in node.children
      ]  #cheeky list comprehension
      #   print(ucb_values)

      # Select the child node with the highest UCB value
      selected_index = ucb_values.index(max(ucb_values))
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

  def expansion(self, leaf_node, valid_moves):
    """ Expand the tree by creating child nodes for the selected leaf node.Assign the leaf node as the parent of each child node."""
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
                          depth=leaf_node.depth + 1)

      # Add the new child node to the children list of the leaf node
      leaf_node.children.append(new_node)

  def make_move(self, move, current_state, whos_move_value):
    ''' What is move (tuple in active or is board state)? When to change to -1 vs 1 for adding in the move (whos move) - is current state a 9x9 or a 3x3x3?'''
    ''' returns new state after move is made '''

    move_x = move[0]
    move_y = move[1]
    current_state[
        move_x,
        move_y] = whos_move_value  #need to multiply by -1 if the current player making the move is the opponent
    #should also consider the active box and valid moves here
    #active box (ignoring if it is already used) create lists of possible places that result in each box
    subbox_x = move_x // 3
    subbox_y = move_y // 3
    subbox = (subbox_x, subbox_y
              )  #check if it is full or not to see if we have all moves]
    new_active_box = (-1, -1)

    #GOES TO THIS BOX
    box22 = [(6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7),
             (8, 8)]
    box21 = [(6, 3), (6, 4), (6, 5), (7, 3), (7, 4), (7, 5), (8, 3), (8, 4),
             (8, 5)]
    box20 = [(6, 0), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2), (8, 0), (8, 1),
             (8, 2)]
    box12 = [(3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7),
             (5, 8)]
    box11 = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4),
             (5, 5)]
    box10 = [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1),
             (5, 2)]
    box02 = [(0, 6), (0, 7), (0, 8), (1, 6), (1, 7), (1, 8), (2, 6), (2, 7),
             (2, 8)]
    box01 = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),
             (2, 5)]
    box00 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1),
             (2, 2)]

    if subbox in box00:
      new_active_box = (0, 0)
    elif subbox in box01:
      new_active_box = (0, 1)
    elif subbox in box02:
      new_active_box = (0, 2)
    elif subbox in box10:
      new_active_box = (1, 0)
    elif subbox in box11:
      new_active_box = (1, 1)
    elif subbox in box12:
      new_active_box = (1, 2)
    elif subbox in box20:
      new_active_box = (2, 0)
    elif subbox in box21:
      new_active_box = (2, 1)
    elif subbox in box22:
      new_active_box = (2, 2)

    #Now we have active box for next move

      #Return 0.2 as a reward

    #Get valid moves in new active box
    new_mini_board = self.pull_mini_board(current_state, new_active_box)
    new_mini_board = self.invert_mini_board(new_mini_board)

    # case: the box is (-1 -1)
    if self.subgame_terminated(new_mini_board) != -3:
      new_active_box = (-1, -1)

    new_valid_moves = self.from_mini_to_big(
        new_mini_board,
        new_active_box)  #new_valid_moves is in terms of the 9x9 matrix

    #Have a list of valid moves in the new mini board

    #Hmmmmm
    return [current_state, new_valid_moves, new_active_box
            ]  # new active is where we play next move, so where did we play

  def from_mini_to_big(self, new_mini_board, new_active_box):
    # print(new_mini_board)
    # print(new_active_box)

    #This allows us to switch between 3x3 and 9x9
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

  def invert_mini_board(self, mini_board):
    ''' the actual board is inverted i.e 0,0 is bottom left corner'''
    new_board = mini_board[::-1, :]
    return new_board

  def pull_mini_board(self, board_state: np.array,
                      mini_board_index: tuple) -> np.array:
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

  #Function is for testing
  def print_tree(self, node, level=0):
    """Prints the tree from the given node."""
    if node is not None:
      print(
          "  " * level +
          f"Move: {node.init_move}, Value: {node.value}, Visits: {node.visits}"
      )
      for child in node.children:
        self.print_tree(child, level + 1)


## For Testing/debugging

# Mock input data
board_dict = {
    'board_state': np.zeros((9, 9)),  # Example of a 9x9 board with all zeros
    'active_box': (1, 1),  # Example of the active box
    'valid_moves': [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4),
                    (5, 5)]  # Example of valid moves
}
# First opponent move
board_dict['board_state'][4, 4] = -1
board_state = board_dict['board_state']

# Instantiate MCTS agent
mcts_agent = Jaspers_MCTS_Agent()

# Test expansion - Print the tree
# root_node = TreeNode(board_dict['board_state'])
# mcts_agent.expansion(root_node, [(1, 1), (2, 2), (3, 3)])  # Example valid moves
# mcts_agent.print_tree(root_node)

# Call move function
selected_move = mcts_agent.move(board_dict)

# Inspect output
# print("Selected move:", selected_move)x
