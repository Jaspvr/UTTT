import numpy as np
import math

class TreeNode:
    def __init__(self, state, parent = None, move=None, active_box=None, valid_moves=None): #pass in state, unsure about parent and move
      self.state = state
      self.parent = parent
      self.init_move = move 
      self.children = [None]*81 #up to 81 (free board with everything open)
      self.visits = 0 #initial is always 0
      self.value = 0 # initial is always 0
      self.active_box = active_box
      self.valid_moves = valid_moves
    #   self.outcome = state.get_outcome() #check and see if the game is over
        

class Jaspers_MCTS_Agent:
    def __init__(self, name: str = 'jaspers_bot', debug=True):
        self.name = name

    def move(self, board_dict: dict) -> tuple:
        board_state = board_dict['board_state'] #9x9 numpy array
        active_box = board_dict['active_box'] #(x, y) tuple of coordinates of the smaller 3x3 game
        valid_moves = board_dict['valid_moves'] #list of all valid moves, i.e list of tuples

        # count = 0
        # while count < 5:

        #Selection phase: Traverse from the root node to a leaf node
        root_state = board_state
        root_node = TreeNode(root_state)
        selected_leaf_node = self.selection(root_node)
        # we now have the leaf node to work with

        #for the root node, selection function will just return back the same node
        self.expansion(selected_leaf_node, valid_moves)

        #to see a couple iterations
        child_index = 0
        child_node = selected_leaf_node.children[child_index]
        self.expansion(child_node, child_node.valid_moves)


        self.print_tree(root_node)


        # print(selected_leaf_node)
        # We now have the selected node to expand on

        # Expansion Phase: our selected node is a leaf node, we have to create its children with all the possible valid moves from that place
        # So get the valid moves for the leaf:
        # Currently no logic for these valid moves, need to probably update the game state associated with the leaf node (leaf node move has been made)
            
        # leaf_valid_moves = self.get_valid_moves(selected_leaf_node)
        # self.expansion(selected_leaf_node, leaf_valid_moves)

        # count += 1


        return valid_moves[0] # placeholder


    def selection(self, node):
        ''' Look at children, go to one with highest ucb, go until reach leaf node, this is the selected node for expansion '''
        # Define the exploration constant
        exploration_constant = 1.41

        # Traverse down the tree until a leaf node is reached
        while all(child is not None for child in node.children):
            # Calculate UCB values for each child node
            ucb_values = [self.calculate_ucb(child, exploration_constant, node.visits) for child in node.children] #cheeky list comprehension

            # Select the child node with the highest UCB value
            selected_index = ucb_values.index(max(ucb_values))
            node = node.children[selected_index]
              
        return node 


    def calculate_ucb(self, node, exploration_constant, parent_visits):
        ''' Calculate the ucb score for selecting the best node during selection process '''
        if node.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        else:
            exploitation_term = node.value / node.visits
            exploration_term = exploration_constant * math.sqrt(math.log(parent_visits) / node.visits)
            return exploitation_term + exploration_term
        
        

    def expansion(self, leaf_node, valid_moves):
        """ Expand the tree by creating child nodes for the selected leaf node.Assign the leaf node as the parent of each child node."""
        for move in valid_moves:
            # Create a new state based on the move
            new_state, new_valid_moves, new_active_box = self.make_move(move, leaf_node.state)

            # Create a new child node with the updated state and link it to the leaf node
            new_node = TreeNode(state=new_state, parent=leaf_node, move=move)

            # Add the new child node to the children list of the leaf node
            leaf_node.children.append(new_node)

    def make_move(self, move, current_state):
        ''' returns new state after move is made '''
        move_x = move[0]
        move_y = move[1]
        current_state[move_x, move_y] = 1 #need to multiply by -1 if the current player making the move is the opponent
        #should also consider the active box and valid moves here
        #active box (ignoring if it is already used) create lists of possible places that result in each box
        subbox_x = move_x // 3
        subbox_y = move_y // 3
        subbox = (subbox_x, subbox_y) #check if it is full or not to see if we have all moves
        #if subbox is full: new_active = (-1,-1)
        # coordinates = [[(i, j) for j in range(col, col + 3) for i in range(row, row + 3)] for row in range(0, 9, 3) for col in range(0, 9, 3)]
        #T
        # possible_boxes = []
        # lb = []
        # for subgame in coordinates:

        box00 = [(0, 0), (0, 3), ]
        box01 = []

        # new_active


        return current_state, new_valid_moves, new_active_box
    
    #Function is for testing
    def print_tree(self, node, level=0):
        """Prints the tree from the given node."""
        if node is not None:
            print("  " * level + f"Move: {node.init_move}, Value: {node.value}, Visits: {node.visits}")
            for child in node.children:
                self.print_tree(child, level + 1)

        

    def get_valid_moves(self, node):
        pass



## For Testing/debugging

        
# Mock input data
board_dict = {
    'board_state': np.zeros((9, 9)),  # Example of a 9x9 board with all zeros
    'active_box': (1, 1),  # Example of the active box
    'valid_moves': [(3,3), (3,4), (3,5), (4,3), (4,5), (5,3), (5,4), (5,5)]  # Example of valid moves
}
# First opponent move
board_dict['board_state'][4, 4] = -1

# Instantiate MCTS agent
mcts_agent = Jaspers_MCTS_Agent()

# Test expansion - Print the tree
# root_node = TreeNode(board_dict['board_state'])
# mcts_agent.expansion(root_node, [(1, 1), (2, 2), (3, 3)])  # Example valid moves
# mcts_agent.print_tree(root_node)

# Call move function
selected_move = mcts_agent.move(board_dict)

# Inspect output
# print("Selected move:", selected_move)x