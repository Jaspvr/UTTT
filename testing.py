# def test1(nothign):
#     return [1, 2, 3]

# a, b, c = test1('something')
# print(a, b, c)

# coordinates = [[[(i, j) for j in range(col, col + 3)] for i in range(row, row + 3)] for row in range(0, 9, 3) for col in range(0, 9, 3)]
# for row in coordinates:
#     print(row)

# print(coordinates)

# Generate 9x9 coordinate matrix
# coordinates = [[(i, j) for j in range(col, col + 3) for i in range(row, row + 3)] for row in range(0, 9, 3) for col in range(0, 9, 3)]

# # Print the coordinates
# for row in coordinates:
#     print(row)

import numpy as np

def check_winner(mini_board):
    # Check rows and columns
    for i in range(3):
        if np.all(mini_board[i, :] == 1) or np.all(mini_board[:, i] == 1):
            return True  # Player 1 wins
        elif np.all(mini_board[i, :] == -1) or np.all(mini_board[:, i] == -1):
            return True  # Player 2 wins

    # Check diagonals
    if np.all(np.diag(mini_board) == 1) or np.all(np.diag(np.fliplr(mini_board)) == 1):
        return True  # Player 1 wins
    elif np.all(np.diag(mini_board) == -1) or np.all(np.diag(np.fliplr(mini_board)) == -1):
        return True  # Player 2 wins

    # Check for a draw
    if np.count_nonzero(mini_board == 0) == 0:
        return True  # Draw

    # If no winner yet
    return False

# Example usage:
mini_board = np.array([[1, -1, 1],
                       [1, 1, -1],
                       [-1, 1, -1]])

result = check_winner(mini_board)
print(result)
