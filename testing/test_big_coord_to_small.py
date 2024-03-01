def get_sub_box(coord):
    # Calculate the row and column indices of the sub-box
    sub_row = coord[0] // 3
    sub_col = coord[1] // 3
    return (sub_row, sub_col)

def map_to_mini_box(move):
    # Get the row and column indices of the move within the mini-box
    mini_row = move[0] % 3
    mini_col = move[1] % 3
    return (mini_row, mini_col)

v = map_to_mini_box((4, 4))
print(v)

v = map_to_mini_box((6, 6))
print(v)
v = map_to_mini_box((0, 0))
print(v)
v = map_to_mini_box((8, 7))
print(v)

# val = get_sub_box((4, 7))
# print(val)

# val = get_sub_box((8, 8))
# print(val)

# val = get_sub_box((0, 0))
# print(val)