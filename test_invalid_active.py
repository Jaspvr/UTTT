# def evil_case()
# # if new_active_box == (-1, -1):
#     all_moves = []
#     tuple_list = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
#     tuples_revised = []
#     for tuple1 in tuple_list:
#     new_mini_board = pull_mini_board(new_state, tuple1)
#     terminated_val = subgame_terminated(new_mini_board)
#     if terminated_val == -3:
#         tuples_revised.append(tuple1)

#     # We now have a revised list of tuples of non finished games
#     for tuple_revised in tuples_revised:
#         # want to get all tuple places on in any of these squares
#         all_moves.append(get_coordinates_in_submatrix(tuple_revised))

#     new_valid_moves = all_moves