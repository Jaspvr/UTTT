import numpy as np
import random

def get_outcome(board_state):
    ''' Big game outcome '''
    #Check for which mini games are over and their result
    #Go through mini games 00 to 22, check if they are over,
    mini_game_tuples = [(i, j) for i in range(3)
                        for j in range(3)]  #List of mini board coordinates
    mini_game_outcomes = []
    for mini_game_tuple in mini_game_tuples:
      mini_game_state = pull_mini_board(board_state, mini_game_tuple)
      mini_game_outcome = subgame_terminated(
          mini_game_state)  #Either 1, 0.5 ,or 0, or -3
      if mini_game_outcome == -3:
        mini_game_outcomes.append(0)
      elif mini_game_outcome == 1:  #win
        mini_game_outcomes.append(1)
      elif mini_game_outcome == 0.5:  #draw
        mini_game_outcomes.append(0)
      elif mini_game_outcome == 0:  #loss
        mini_game_outcomes.append(-1)

    #Make the mini game outcomes into a 3x3 np.array instead of a list
    # Put the result in a new 3x3 representing the big board
    mini_game_outcomes_matrix = np.array(mini_game_outcomes).reshape(3, 3)
    print(mini_game_outcomes_matrix)
    #Check big game
    big_game_outcome = subgame_terminated(mini_game_outcomes_matrix)
    return big_game_outcome

def pull_mini_board(board_state: np.array,
                      mini_board_index: tuple) -> np.array:
    ''' extracts a mini board from the 9x9 given the its index'''
    temp = board_state[mini_board_index[0] * 3:(mini_board_index[0] + 1) * 3,
                       mini_board_index[1] * 3:(mini_board_index[1] + 1) * 3]
    return temp

def subgame_terminated(mini_board):
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

    # # Check for a draw
    # if np.count_nonzero(mini_board == 0) == 0:
    #   return 0.5  # Draw
    # DOESNT WORK SINCE THERE COULD BE NONZEROS AND HAVE IT STILL

    #Check if is it possible to make a line with either 1s and 0s or with -1s and 0s. If not, its a draw
    # Check if lines can still be formed by either (1s and 0s) or (-1s and 0s)
    for i in range(3):
        # Check rows and columns
        if (0 in mini_board[i, :]) and (1 in mini_board[i, :]) and (-1 not in mini_board[i, :]):
            return -3
        if (0 in mini_board[:, i]) and (1 in mini_board[:, i]) and (-1 not in mini_board[:, i]):
            return -3
        if (0 in mini_board[i, :]) and (-1 in mini_board[i, :]) and (1 not in mini_board[i, :]):
            return -3
        if (0 in mini_board[:, i]) and (-1 in mini_board[:, i]) and (1 not in mini_board[:, i]):
            return -3

    # Check diagonals
    diagonal1 = np.diag(mini_board)
    diagonal2 = np.diag(np.fliplr(mini_board))
    if (0 in diagonal1) and (1 in diagonal1) and (-1 not in diagonal1):
        return -3
    if (0 in diagonal1) and (-1 in diagonal1) and (1 not in diagonal1):
        return -3
    if (0 in diagonal2) and (1 in diagonal2) and (-1 not in diagonal2):
        return -3
    if (0 in diagonal2) and (-1 in diagonal2) and (1 not in diagonal2):
        return -3

    # must be draw
    return 0.5


array = np.array( [
    [ 1, -1,  1, -1,  1,  1, -1,  0,  0],
    [ 1,  0,  0,  0, -1, -1, -1,  0,  1],
    [ 1,  0,  1,  0,  0, -1, -1, -1,  0],
    [-1, -1,  1,  0,  1,  1, -1,  1,  1],
    [-1,  1,  1, -1, 1, -1,  1, -1, -1],
    [-1,  0,  0,  1, -1,  1,  1, 1,  1],
    [-1, -1,  1,  0,  1,  1, -1,  1, -1],
    [ 1, -1, -1,  1,  0, -1,  0,  1,  0],
    [-1,  1,  1,  1,  1,  1, -1, -1, -1]
] )

array2 = np.array( [
    [ -1, 1,  1],
    [ 1,  -1,  -1,],
    [ 0,  1,  1],
] )

o = subgame_terminated(array2)
print(o)