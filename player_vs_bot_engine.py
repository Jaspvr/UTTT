# Import the bots and neccessary libraries:
from random_bot import Random_Bot
from mcts_bot import Jaspers_MCTS_Agent

import numpy as np
import random
import math


class bvb_engine:
    def __init__(self, name: str = 'bvb engine'):
        self.name = name

    def simulate_bot_game(self, board_dict):
        ''' Simulate a game between two bots given the initial board state'''
        # Create instances of bot
        mcts_agent = Jaspers_MCTS_Agent()

        # Get user's name
        players_name = input("Enter your name: ")

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
            selected_move = self.get_player_move(board_dict)

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
            winner = players_name
        elif game_result == 1:
            winner = mcts_agent.name
        else:
            winner = "Draw!"

        # Return the winner and the final board state
        return winner, board_dict['board_state']
    
    def get_player_move(self, board_dict):
        # Print board state to the user with a list of their valid moves
        board_state = board_dict['board_state']
        valid_moves = board_dict['valid_moves']

        print("Current board state:")
        print(board_state)
        print("Your valid moves:")
        print(valid_moves)

        # Get user inputted tuple in the correct format (provide message if format is incorrect)
        user_input = list(input("Enter one of your valid moves in 00 format: "))
        user_input[0], user_input[1] = int(user_input[0]), int(user_input[1])
        user_input = tuple(user_input)
        
        # Return their move tuple
        return user_input



# Create an initial game state to feed to a bot
board_dict = {
    'board_state': np.zeros((9, 9)),  # Initial board
    'active_box': (1, 1),  # Initial active box
    'valid_moves': [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4),(5, 5), (4, 4)]  # Initial valid moves
}

# Create instance of the engine
game_engine = bvb_engine()

# Run the game
winner, end_state = bvb_engine.simulate_bot_game(game_engine, board_dict)
print(end_state)
print("The winner is: ")
print(winner)

