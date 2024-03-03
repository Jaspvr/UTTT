# Import the bots and neccessary libraries:
import sys
print(sys.path)
sys.path.append('/Users/jasperhalvorson/p/UTTT/')
from ..bots.random_bot import Random_Bot
from ..bots.mcts_bot import Jaspers_MCTS_Agent

import numpy as np


class bvb_engine:
    def __init__(self, name: str = 'bvb engine'):
        self.name = name

    def simulate_bot_game(self, board_dict):
        ''' Simulate a game between two bots given the initial board state'''
        # Create instances of bots
        random_agent = Random_Bot()
        mcts_agent = Jaspers_MCTS_Agent()

        game_result = None
        while True:
            # Get move from the first bot 
            selected_move = mcts_agent.move(board_dict)

            # Update the board dictionary
            new_state, new_valid_moves, new_active_box = mcts_agent.make_move(selected_move, board_dict['board_state'], 1)
            board_dict['board_state'] = new_state
            board_dict['active_box'] = new_active_box
            board_dict['valid_moves'] = new_valid_moves

            # Check if anyone has won
            game_result = mcts_agent.get_outcome(board_dict['board_state'])
            if game_result != -3:
                break

            # Get move from the second bot 
            selected_move = random_agent.move(board_dict)

            # Update the board dictionary and check if anyone has won
            new_state, new_valid_moves, new_active_box = mcts_agent.make_move(selected_move, board_dict['board_state'], -1)
            board_dict['board_state'] = new_state
            board_dict['active_box'] = new_active_box
            board_dict['valid_moves'] = new_valid_moves

            # Check if anyone has won
            game_result = mcts_agent.get_outcome(board_dict['board_state'])
            if game_result != -3:
                break
            
        # Get the winner
        winner = None
        if game_result == 0:
            winner = random_agent.name
        elif game_result == 1:
            winner = mcts_agent.name
        else:
            winner = "Draw!"

        # Return the winner and the final board state
        return winner, board_dict['board_state']

    def format_board(self, board):
        formatted_board = ''
        for i in range(len(board)):
            if i % 3 == 0 and i != 0:
                formatted_board += '-' * 25 + '\n'
            for j in range(len(board[i])):
                if j % 3 == 0 and j != 0:
                    formatted_board += '| '
                formatted_board += ' '
                if board[i][j] == 1:
                    formatted_board += 'X '
                elif board[i][j] == -1:
                    formatted_board += 'O '
                else:
                    formatted_board += '. '
            formatted_board += '\n'
        return formatted_board
    


# Create an initial game state to feed to a bot
board_dict = {
    'board_state': np.zeros((9, 9)),  # Initial board
    'active_box': (1, 1),  # Initial active box
    'valid_moves': [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4),(5, 5), (4, 4)]  # Initial valid moves
}

# Create instance of the engine
game_engine = bvb_engine()

wins = 0
i = 0
while i < 1:
    # Run the game
    winner, end_state = bvb_engine.simulate_bot_game(game_engine, board_dict)
    if winner == "mcts_bot":
        wins+=1
        # print("W")
    else:
        print("L")
    i+=1

print(game_engine.format_board(end_state))
print("The winner is: ")
print(winner)

