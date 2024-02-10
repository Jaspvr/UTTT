user_input = list(input("Enter one of your valid moves in 00 format: "))
user_input[0], user_input[1] = int(user_input[0]), int(user_input[1])
user_input = tuple(user_input)


print(user_input)