import numpy as np
import math
import random
''' core imports '''
import numpy as np
import matplotlib.pyplot as plt

''' development imports'''
from time import perf_counter
from tqdm import tqdm

''' visualization imports '''
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('mycmap', ['lightgrey', 'white'])
import matplotlib.colors as mcolors
tab10_names = list(mcolors.TABLEAU_COLORS) # create a list of colours

def checkerboard(shape):
    # from https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    return np.indices(shape).sum(axis=0) % 2


class TreeNode:
  def __init__(self,
               state,
               parent=None,
               move=None,
               active_box=None,
               valid_moves=None,
               visits=None,
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
    
    root_state = board_state
    root_node = TreeNode(root_state,
                          active_box=active_box,
                          valid_moves=valid_moves,
                          visits=1,
                          depth=0, 
                          )

    count = 0
    while count < 10:

      #Selection phase: Traverse from the root node to a leaf node
      selected_leaf_node = self.selection(
          root_node
      )  # we now have the leaf node to work with. for the root node, selection function will just return back the same node

      # Expansion Phase: our selected node is a leaf node, we have to create its children with all the possible valid moves from that place
      flag = False
      if(selected_leaf_node.visits != 0): #Account for case of not visited the node before
        self.expansion(selected_leaf_node, selected_leaf_node.valid_moves
                      )  #We have now expanded. We should simulate
      else: 
        selected_leaf_node.visits += 1
        flag = True

      # self.print_tree(root_node) ##LOOKS GOOD FOR NOW UP TO HERE

      # Simulation Phase: simulate down the tree until terminal state is reached
      reward = self.simulation(
          selected_leaf_node
      )  #return the value of the game end (win, draw, loss)
      # print(reward)
      # if flag:
      #   selected_leaf_node.visits -= 1

      #WORKS UP TO HERE

      # Backpropogate: Propogate ___ up the tree for node UCB scores (i think)
      self.backpropogate(selected_leaf_node, reward)
      count += 1
    
    # self.print_tree(root_node)
    # Find the best move (highest number of visits? )
    max_value = -1
    max_child = None
    for child in root_node.children:
      if child.visits > max_value:
        max_child = child
  
    move_to_make = tuple(max_child.init_move)

    return move_to_make  # placeholder

  def backpropogate(self, selected_leaf_node, reward):
    #BACKPROPOGATE, adding outcome and visit count up

    while selected_leaf_node.parent is not None:
      #naming here is not consistent; selected_leaf_node, but after first while iteration, not a leaf node, since backpropogated.
      selected_leaf_node.value += reward
      selected_leaf_node.visits += 1
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
    while True:  
      if self.get_outcome(board_state) != -3: #-3 is game is not over
        break

      # if(count > 20):
      #   break

      #randomly play one move
      # print(board_state, valid_moves)
      move = random.choice(valid_moves)
    
      #update board state, valid moves, active box with make_move
      board_state, valid_moves, active_box = self.make_move(
          move, board_state, whos_move_value)

      #update whos turn
      whos_move_value *= -1
      count+=1

    # Now we have reached a terminal state, set the value of the game to the leaf node
    outcome_value = self.get_outcome(board_state)
    # print(board_state)
    count1 = 0
    count2 = 0
    for i in board_state:
      for j in i:
        if j == 1:
          count1 +=1
        if j == -1:
          count2 += 1
    # print(count1, count2)
    return outcome_value

  def move_leads_to_a_finished_game(self, move, board_state):
    ''' Check if the move leads to a finished game '''
    new_active_box = self.active_box_after_move(move, board_state)
    #if that box is done. return True
    if self.subgame_terminated != -3:  # game is done
      return True

    return False

  def active_box_after_move(self, move, board_state):
    ''' returns the active box after the move is made '''
    #GOES TO THIS BOX

    subbox = move

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
  def selection(self, node):  #QJKAPROVE #Approve
    ''' Look at children, go to one with highest ucb, go until reach leaf node, this is the selected node for expansion '''
    # Define the exploration constant
    exploration_constant = 1.41

    # Traverse down the tree until a leaf node is reached
    while not all(child is None for child in
                  node.children):  #at a leaf node when all children are None
      # Calculate UCB values for each child node
      # if node.visits==0:
      #   break
      ucb_values = [
          self.calculate_ucb(child, exploration_constant, node.visits)
          for child in node.children
      ]  #cheeky list comprehension

      # Select the child node with the highest UCB value
      selected_index = ucb_values.index(max(ucb_values))
      node = node.children[selected_index]

    return node

  #approved:
  def calculate_ucb(self, node, exploration_constant, parent_visits):
    ''' Calculate the ucb score for selecting the best node during selection process '''
    if node.visits == 0:
      return float('inf')  # Prioritize unvisited nodes
    else:
      exploitation_term = node.value / node.visits
      # print(parent_visits, node.visits)
      exploration_term = exploration_constant * math.sqrt(
          math.log(parent_visits) / node.visits)
      return exploitation_term + exploration_term


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
      # print(new_node.state)

    # print(new_node.state)

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

    #Now we have active box for next move

    # case: the box is (-1 -1)
    all_moves = []
    # get miniboard for our tuple
    # print(new_state, new_active_box, move)
    new_mini_board = self.pull_mini_board(new_state, new_active_box)

    if self.subgame_terminated(new_mini_board) != -3:
      new_active_box = (-1, -1)

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
        # want to get all tuple places on in any of these squares
        all_moves.append(self.get_coordinates_in_submatrix(tuple_revised))

      #Check for spaces of '0'. these are our valid moves
      new_valid_moves = []
      for move_list in all_moves:
        for move_tuple in move_list:
          if new_state[move_tuple[0], move_tuple[1]] == 0:
            new_valid_moves.append(move_tuple)

    
      # new_valid_moves = all_moves
    #   print(new_valid_moves)
      return [new_state, new_valid_moves, new_active_box]  # new active is where we play next move, so where did we play

      #Return 0.2 as a reward
    else:
        #Get valid moves in new active box
        new_mini_board = self.pull_mini_board(current_state, new_active_box)

        new_valid_moves = self.from_mini_to_big(
            new_mini_board,
            new_active_box)  #new_valid_moves is in terms of the 9x9 matrix

        #Have a list of valid moves in the new mini board
        return [new_state, new_valid_moves, new_active_box] 
    
  def get_coordinates_in_submatrix(self, coord_tuple):
    submatrix_coordinates = []
    for row in range(3):
        for col in range(3):
            submatrix_coordinates.append((coord_tuple[0]*3 + row, coord_tuple[1]*3 + col))
    return submatrix_coordinates

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

class random_bot:
    '''
    this bot selects a random valid move
    '''
    def __init__(self, name = 'beep-boop'):
        self.name = name
    def move(self, board_dict):
        random_index = np.random.choice(len(board_dict['valid_moves']))
        return board_dict['valid_moves'][random_index]

class uttt_engine():
    def __init__(self):
        self.active_box = (-1,-1)    # (-1,-1) means any box can be played in
        self.board_state = np.zeros((9,9))
        self.finished_boxes = np.zeros((3,3)) # 1 for agent1, -1 for agent2. "6" indicates stalemate
        self.finished = False
        self.finished_win = False
        self.finished_stale = False
        self.current_player = 1 # starting player
        self.game_log = ''

    ''' -------------- Initialization ------'''
    def load_agents(self, agent1: bot_template, agent2: bot_template) -> None:
        ''' agent1 and agent2 are uttt agents '''
        self.agents = [agent1, agent2]

    ''' --------------- Logging ----------- '''

    def get_game_log(self) -> str:
        ''' returns the current game encoding (for visualization) '''
        return self.game_log

    def log_move(self, position: tuple) -> None:
        # convert tuple to string, append character to existing game code
        offset = 32
        self.game_log += (chr(position[0]*9 + position[1] + offset))

    def load_game(self, game_string:str) -> None:
        ''' assumes the game string is valid (this is not strictly enforced) '''
        for encoded_position in game_string:
            # update game board
            self.move(self.unconvert(encoded_position), log=False)

    def unconvert(self, symbol):
        ''' part of loading a game from a string '''
        offset = 32
        int_position = ord(symbol) - offset
        return (int_position//9, int_position%9)

    ''' ------------- Logic --------------- '''

    def get_query_dict(self) -> dict:
        ''' can be used in development to understand what the bot is being provided for logic '''
        return {'board_state': self.board_state * self.current_player, # *current_player for changing -1 to 1 and vice versa
                'active_box':  self.active_box,
                'valid_moves': self.get_valid_moves()}

    def query_player(self, loud: bool = False) -> None:
        '''
        send a request to a player instance for a move
        updates the game board with the desired move
        if no valid move is returned, a random move is played for that agent
        the "loud" argument can be used to hear/silence warnings
        '''
        # check agents are loaded
        if not hasattr(self, 'agents'):
            print('must load agents')
            return

        # check game is not finished
        if self.finished:
            print('no valid moves in terminal state')
            #print("(something went wrong, you shouldn't be here)")
            #print('put in a "game_finished" check somewhere')
            return

        # send the request with board information in the form of a dictionary
        temp_valid_moves = self.get_valid_moves()
        temp_valid_moves = [tuple(x) for x in temp_valid_moves]

        try:
            desired_move = tuple(self.agents[0].move(self.get_query_dict()))
            if desired_move not in temp_valid_moves:
                random_index = np.random.choice(np.arange(len(temp_valid_moves)))
                desired_move = tuple(temp_valid_moves[random_index])
                if loud:
                    print(f'warning: {self.agents[0].name} played an invalid move. converting to random valid alternative')
            # update board
            self.move(position = desired_move)

        except:
            # shouldn't get here, but this chunk of code exists for safety
            if loud:
                print(f'warning: exception raised in "query" for {self.agents[0].name}')
            random_index = np.random.choice(np.arange(len(temp_valid_moves)))
            # desired_move = tuple(temp_valid_moves[random_index])

            # update board
            self.move(position = desired_move)

    def switch_active_player(self) -> None:
        ''' switch the current player value and the agent list '''
        # this is called at the end of .move()
        self.agents = self.agents[::-1]
        self.current_player *= -1

    def getwinner(self) -> int:
        ''' new method
        returns the integer indicating the winning player
        (subject to change)
        '''
        if self.finished:
            if self.finished_win:
                return self.current_player
            else:
                return 0

    def check_validity(self, position: tuple) -> bool:
        ''' check whether position - a tuple - is valid '''
        box_validity = (self.active_box == self.map_to_major(position)) \
                        or (self.active_box == (-1,-1))
        open_validity = (self.board_state[position] == 0)
        return box_validity and open_validity

    def check_line(self, box: np.array) -> bool:
        '''
        box is a (3,3) array (typically a mini-board)
        returns True if a line is found
        '''
        for i in range(3):
            if abs(sum(box[:,i])) == 3: return True # horizontal
            if abs(sum(box[i,:])) == 3: return True # vertical

        # diagonals
        if abs(box.trace()) == 3: return True
        if abs(np.rot90(box).trace()) == 3: return True

    def map_to_major(self, position: tuple) -> tuple:
        '''
        converts position to major coordinates
        eg: (5,3) -> (1,1)
        '''
        return(position[0]//3, position[1]//3)

    def map_to_minor(self, position: tuple) -> tuple:
        '''
        converts position into mini coordinates
        eg: (5,3) -> (2,0)
        '''
        return (position[0]%3, position[1]%3)

    def check_full_stale(self) -> None:
        ''' this might be impossible? '''
        # get number of invalid boxes

        if (self.finished_boxes == 0).sum() == 0:
            self.finished_stale = True
            self.finished = True

    def move(self, position: tuple, log: bool = True) -> None:
        '''
        the main game logic. board updates and logic checks.
        '''
        if self.finished:
            print('no move played, game is finished')
            return

        if self.check_validity(position):

            # log move
            if log:
                self.log_move(position)

            # place marker
            self.board_state[position] = self.current_player

            # select both scales
            temp_box = self.map_to_major(position)
            temp_minor_box = self.board_state[3*temp_box[0]:3*temp_box[0]+3,
                                              3*temp_box[1]:3*temp_box[1]+3]

            ''' check line at minor scale '''
            if self.check_line(temp_minor_box):
                self.finished_boxes[self.map_to_major(position)] = self.current_player

                # check line at major scale
                if self.check_line(self.finished_boxes):
                    self.finished_win = True
                    self.finished = True
                    return # end the whole thing immediately (will cause stalemate bug without this !)

            # if no squares are open, mark as stale
            elif (temp_minor_box == 0).sum() == 0:
                self.finished_boxes[self.map_to_major(position)] = 6 # indicates stalemate in that box

            ''' is the whole game board stale? '''
            # if it's stale, set the appropriate flags
            self.check_full_stale()

            ''' calculate active box '''
            self.active_box = self.map_to_minor(position)
            # if that box is won or stale flag it
            if self.finished_boxes[self.active_box] != 0:
                self.active_box = (-1,-1)

            # switch player
            self.switch_active_player()

    def get_valid_moves(self) -> np.array:
        '''
        returns an array (N,2) of valid moves
        '''

        if self.finished:
            print('no valid moves in terminal state')
            return np.empty(0)
        # define masks that cover the board
        # across the whole board
        full_board_mask = (self.board_state == 0)
        # active square
        active_box_mask = np.zeros((9,9),dtype=bool)
        # identifies finished major boxes
        a = np.repeat(self.finished_boxes,3).reshape(3,9)
        b = np.tile(a,3).reshape(9,9)
        finished_box_mask = (b == 0)

        if self.active_box == (-1,-1):
            active_box_mask[:] = True
            active_box_mask *= finished_box_mask
        else:
            active_box_mask[3*self.active_box[0]:3*self.active_box[0]+3,
                            3*self.active_box[1]:3*self.active_box[1]+3] = True

        # return get union of maps
        return np.array(np.where(active_box_mask * full_board_mask)).T

    ''' ------------- Visualization ------- '''

    def draw_valid_moves(self) -> None:
        ''' visualization tool
        plots the valid moves as purple squares
        to be called after the .draw_board() method
        '''
        moves = self.get_valid_moves()
        plt.scatter(moves[:,0],moves[:,1],marker='s',c='purple',alpha=0.3, s=50)

    def draw_board(self, marker_size: int = 100, ticks: str = 'off') -> None:
        ''' visualization tool
        plots a checkerboard and markers for all plays.
        lines distinguish mini-boards and finished boards are coloured in
        '''
        plt.imshow(checkerboard((9,9)), cmap=cmap, origin='lower')
        for i in [-0.5,2.5,5.5, 8.5]:
            plt.axvline(i,c='k')
            plt.axhline(i,c='k')

        if ticks == 'off':
            plt.axis('off')
        else:
            plt.xticks(np.arange(9))

        plt.scatter(*np.where(self.board_state == -1), marker='x', s=marker_size, c='tab:blue')
        plt.scatter(*np.where(self.board_state == 1),  marker='o', s=marker_size, c='tab:orange')

        x_boxes = np.where(self.finished_boxes == -1)
        o_boxes = np.where(self.finished_boxes == 1)
        plt.scatter(x_boxes[0]*3+1,x_boxes[1]*3+1,marker='s',s=marker_size*50,alpha=0.6,c='tab:blue')
        plt.scatter(o_boxes[0]*3+1,o_boxes[1]*3+1,marker='s',s=marker_size*50,alpha=0.6,c='tab:orange')

        stale_boxes = np.where(self.finished_boxes == 6)
        plt.scatter(stale_boxes[0]*3+1, stale_boxes[1]*3+1, marker='s', s=marker_size*50, alpha=0.3,c='k')

# initialize the engine
engine = uttt_engine()

# load two agents into the game
# We will be overriding some bot logic
engine.load_agents(random_bot(name = 'the bot known as player 1'),
                   random_bot(name = 'the second bot'))
