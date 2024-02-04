def get_coordinates_in_submatrix(coord_tuple):
    submatrix_coordinates = []
    for row in range(3):
        for col in range(3):
            submatrix_coordinates.append((coord_tuple[0]*3 + row, coord_tuple[1]*3 + col))
    return submatrix_coordinates

# Test
coord_tuple = (0, 1)
result = get_coordinates_in_submatrix(coord_tuple)
print(result)