import numpy as np
import random
import math

def make_move(move, current_state, whos_move_value): #Valid moves are wrong! need to be updated in terms of the actual grid
    ''' current state is a 9x9, move is a tuple, whos_move_value is either 1 or -1 '''
    ''' returns new state after move is made '''
    # Get the new state after the move
    move_x = move[0]
    move_y = move[1]
    new_state = current_state
    new_state = np.copy(current_state)
    new_state[move_x, move_y] = whos_move_value #set the square to value of current player

    # Get the active box

    #all the values actually in each box
    # box22 = [(6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7),(8, 8)]
    # box21 = [(6, 3), (6, 4), (6, 5), (7, 3), (7, 4), (7, 5), (8, 3), (8, 4),(8, 5)]
    # box20 = [(6, 0), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2), (8, 0), (8, 1),(8, 2)]
    # box12 = [(3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7),(5, 8)]
    # box11 = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4),(5, 5)]
    # box10 = [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1),(5, 2)]
    # box02 = [(0, 6), (0, 7), (0, 8), (1, 6), (1, 7), (1, 8), (2, 6), (2, 7),(2, 8)]
    # box01 = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),(2, 5)]
    # box00 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1),(2, 2)]

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
    new_mini_board = pull_mini_board(new_state, new_active_box)

    if subgame_terminated(new_mini_board) != -3:
      new_active_box = (-1, -1)

    if new_active_box == (-1, -1):
      tuple_list = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
      tuples_revised = []
      for tuple1 in tuple_list:
        new_mini_board = pull_mini_board(new_state, tuple1)
        terminated_val = subgame_terminated(new_mini_board)
        if terminated_val == -3:
          tuples_revised.append(tuple1)

      # We now have a revised list of tuples of non finished games
      for tuple_revised in tuples_revised:
        # want to get all tuple places on in any of these squares
        all_moves.append(get_coordinates_in_submatrix(tuple_revised))
    
      new_valid_moves = all_moves
    #   print(new_valid_moves)
      return [new_state, new_valid_moves, new_active_box]  # new active is where we play next move, so where did we play

      #Return 0.2 as a reward
    else:
        #Get valid moves in new active box
        new_mini_board = pull_mini_board(current_state, new_active_box)
        new_mini_board = invert_mini_board(new_mini_board)

        new_valid_moves = from_mini_to_big(
            new_mini_board,
            new_active_box)  #new_valid_moves is in terms of the 9x9 matrix

        #Have a list of valid moves in the new mini board
        return [new_state, new_valid_moves, new_active_box]  
    
def pull_mini_board(board_state: np.array, mini_board_index: tuple) -> np.array:
    ''' extracts a mini board from the 9x9 given the its index'''
    temp = board_state[mini_board_index[0] * 3:(mini_board_index[0] + 1) * 3,
                       mini_board_index[1] * 3:(mini_board_index[1] + 1) * 3]
    return temp

def subgame_terminated(mini_board): #No function calls within
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

def get_coordinates_in_submatrix(coord_tuple):
    submatrix_coordinates = []
    for row in range(3):
        for col in range(3):
            submatrix_coordinates.append((coord_tuple[0]*3 + row, coord_tuple[1]*3 + col))
    return submatrix_coordinates

def invert_mini_board(mini_board):
    ''' the actual board is inverted i.e 0,0 is bottom left corner'''
    new_board = mini_board[::-1, :]
    return new_board

def from_mini_to_big(new_mini_board, new_active_box):
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

board_dict = {
    'board_state': np.zeros((9, 9)),  # Example of a 9x9 board with all zeros
    'active_box': (1, 1),  # Example of the active box
    'valid_moves': [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4),
                    (5, 5)]  # Example of valid moves
}
# Preset first opponent move
board_dict['board_state'][4, 4] = -1
# current_state = board_dict['board_state']

# Define your values for the 9x9 array
current_state = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

whos_move_value = 1
move1 = (3, 3)

# a = pull_mini_board(current_state, move1)

# print(a)


# new_state, new_valid_moves, new_active_box = make_move(move1, current_state, whos_move_value)
# print(new_state)
# print(new_valid_moves)
# print(new_active_box)

# count = 0
# while(count<20):
#     n = 1
#     if (count % 2) == 0:
#        n = -1
#     new_state, new_valid_moves, new_active_box = make_move(random.choice(new_valid_moves), new_state, whos_move_value*n)
#     print(new_state)
#     print(new_valid_moves)
#     print(new_active_box)
#     count+=1


# From mini to big
