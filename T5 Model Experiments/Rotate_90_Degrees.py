import json

def rotate_grid_90(grid, Prefix_on = True):
    prefix = ""
    grid = grid.split(" ")

    if Prefix_on == True:
        prefix = grid[0:3]
        grid = grid[3:]

    # Transpose the grid
    transposed = list(zip(*grid))
    
    # Reverse each row in the transposed grid to complete the 90 degree rotation
    rotated = ["".join(reversed(row)) for row in transposed]
    string_grid = ' '.join(rotated)

    if Prefix_on == True:
        return prefix[0] + " "+ prefix[2] + " " + prefix[1] + " " + string_grid
    else: 
        return string_grid
    
filename = 97999447

with open(f'Tasks/prefixed_str_{filename}.json', 'r') as file:
    data = json.load(file)

final_json = {}
new_train_data = []
for task in data["train"]:

    new_dict = {}
    new_dict["input"] = rotate_grid_90(task["input"])
    new_dict["output"] = rotate_grid_90(task["output"])
    new_train_data.append(new_dict)

final_json["train"] = new_train_data

new_test_data = []
for task in data["test"]:

    new_dict = {}
    new_dict["input"] = rotate_grid_90(task["input"])
    new_dict["output"] = rotate_grid_90(task["output"])
    new_test_data.append(new_dict)

final_json["test"] = new_test_data

with open(f'Tasks/rotated_prefixed_str_{filename}.json', 'w') as file:
    json.dump(final_json, file, indent=4)
        
    