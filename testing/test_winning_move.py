win_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    
    # Check if a player has won
def check_winner(player):
    for combo in win_combinations:
        if all(board[i] == player for i in combo):
            return True
    return False


def winning_moves(board):
    
    # Check winning moves for 1s (our moves)
    winning_moves_list = []
    for i, val in enumerate(board):
        if val == 0:
            board[i] = 1
            if check_winner(1):
                winning_moves_list.append(i)
            board[i] = 0
    
    return winning_moves_list

# Example usage:
board = [1, 0, 1, 0, 1, -1, 0, -1, 0]
print("Winning moves:", winning_moves(board))
