import numpy as np
import random


class Random_Bot: 
    def __init__(self, name: str = 'jaspers_bot', debug=True):
        self.name = name

    def move(self, board_dict) -> tuple:
        valid_moves = board_dict['valid_moves']
        return random.choice(valid_moves)


#Quick test:
# board_dict = {
#     'board_state': np.zeros((9, 9)),  # Example of a 9x9 board with all zeros
#     'active_box': (1, 1),  # Example of the active box
#     'valid_moves': [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4),
#                     (5, 5)]  # Example of valid moves
# }
# # First opponent move
# board_dict['board_state'][4, 4] = -1
# board_state = board_dict['board_state']

# # Instantiate MCTS agent
# random_agent = Random_Bot()

# # Call move function
# selected_move = random_agent.move(board_dict)
# print(selected_move)