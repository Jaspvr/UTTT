# UTTT
Ultimate Tic Tac Toe is a complex variation of tic tac toe involving 1 large board of 9 subgames of tic tac toe. More info on the game rules can be found here: https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe. This repository contains an AI to optimally play UTTT.

# Monte Carlo Tree Search
The UTTT bot, mcts_bot, in this repo uses a Monte Carlo Search Tree to find the most optimal move.

When the UTTT bot is called it is passed a game state that it then represents as the root node of the game tree. Each valid move from this root state are then represented by children nodes of the root node, each valid move from the children states are represented as children of these nodes, and so on.

Based on the following 4-step search strategy, the bot will select a valid move to send back to the engine: 

Selection: Select a random leaf node (in our case this represents a game state)
Expansion: Create subsequent child nodes on the leaf node for each valid move from this state
Simulation: Simulate a random sequence of moves from each 'player' until a terminal state is reached
Backpropagation: Propogate results back up the tree (wins, draws, losses) to contribute to the rating of the moves that were taken to lead to the end state

The more iterations of this search, the stronger a move the bot will make
<img width="516" alt="image" src="https://github.com/Jaspvr/UTTT/assets/114035580/4a25a54c-d819-4edc-b6e5-19912de0d506">



# Game engines / How to see the bot in action
- The bot vs player game engine can be run to play the MCTS bot in the command line
- The bot vs bot game engine can be run to see the result between a random moves bot and the MCTS bot over several games - other bots could be added in.

There is currently no UI for playing this AI - coming soon!
