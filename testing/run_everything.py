import itertools


def generate_board_configurations():
  empty_board = [0] * 9
  return list(itertools.product([-1, 0, 1], repeat=9))

def generate_board_configurations_with_a_win():
  board_configurations = generate_board_configurations()
  dataset = []

  # For every possible board, 
  for board in board_configurations:
      # Find if there is winning moves for this board
      label = determine_winning_moves(board)
      if label != -1:
        dataset.append(board)

  print(dataset)
         
  
  # all_boards = generate_board_configurations()
  # new_boards = []
  # for board in all_boards:
  #     if(determine_winning_moves(board)!=-1):
  #         new_boards.append(board)

def determine_winning_moves(board):
  board = list(board)
  # Check winning moves for 1s (our moves)
  winning_moves_list = []
  for i, val in enumerate(board):
      if val == 0:
          board[i] = 1
          if check_winner(1):
              winning_moves_list.append(i)
          board[i] = 0
  
  if(len(winning_moves_list)==0):
      return -1
  return winning_moves_list


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


o = generate_board_configurations_with_a_win()
print(o)