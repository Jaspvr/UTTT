import numpy as np

def pull_mini_board(board_state: np.array,
                      mini_board_index: tuple) -> np.array:
    ''' extracts a mini board from the 9x9 given the its index'''
    temp = board_state[mini_board_index[0] * 3:(mini_board_index[0] + 1) * 3,
                       mini_board_index[1] * 3:(mini_board_index[1] + 1) * 3]
    return temp

current_state = np.zeros((9, 9))
current_state[0, 0] = 1
current_state[0, 1] = 2
current_state[0, 2] = 3
current_state[1, 0] = 4
current_state[1, 1] = 5
current_state[1, 2] = 6

print(current_state)
new_mini_board = pull_mini_board(current_state, (0, 0))
print(new_mini_board)
# new_mini_board = np.array([[0, 1, 0],
#  [0, -1, 1],
#  [-1, 0, 1]])

# new_valid_moves = []

# for row in new_mini_board:
#     for col in row:
#         if col == 0:
#           new_valid_moves.append((row, col))

# print(new_valid_moves)