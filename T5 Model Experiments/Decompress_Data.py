
import json 
from utilities import string_to_lists
from collections import Counter
from typing import List
import numpy as np #type: ignore

# Load ARC data
with open("Tasks/31-STR-ARC/22eb0ac0_31_hor_str.json", 'r') as file:
    data = json.load(file)

def rotate_90_clockwise(matrix):
    # Zip the rows and reverse each row to rotate the matrix 90 degrees clockwise
    return [list(reversed(col)) for col in zip(*matrix)]

def all_equal(lst):
    return all(x == lst[0] and x != 0 for x in lst)


def rotate_str_90(grid, Prefix_on = True):
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




def most_common_number(lists: List[List[int]]) -> int:
    flat_list = [num for sublist in lists for num in sublist]
    counter = Counter(flat_list)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None


def is_hor(grids_dict):
    most_common_num = most_common_number(string_to_lists(grids_dict["input"])[3:])
    output = string_to_lists(grids_dict["output"])[3:]
    for x in output:
        if x[0] != most_common_num and all(element == x[0] for element in x):
            return True
    rotated = np.array(output).transpose().tolist()
    for x in rotated:
        if x[0] != most_common_num and all(element == x[0] for element in x):
            return False
    return -1


    
def check_which(data,flag = "train"):
    counter_hor = 0
    counter_ver = 0
    for task in data[flag]:
        if is_hor(task) == True:
            counter_hor += 1
        if is_hor(task) == False:
            counter_ver +=1
        
    print("Horizontal",counter_hor,"Vertical",counter_ver)

check_which(data)
check_which(data,flag="test")


def maker(data):
    full_hor = {"train":[],"test":[]}
    counter = 0
    for task in data["train"]:
        if  counter < 7 and is_hor(task) == False:
            new_dict = {}
            new_dict["input"] = rotate_str_90(task["input"])
            new_dict["output"] = rotate_str_90(task["output"])
            counter += 1
            full_hor["train"].append(new_dict)
        else:
            full_hor["train"].append(task)
    counter = 0
    for task in data["test"]:
        if  counter < 11 and is_hor(task) == True:
            new_dict = {}
            new_dict["input"] = rotate_str_90(task["input"])
            new_dict["output"] = rotate_str_90(task["output"])
            counter += 1
            full_hor["test"].append(new_dict)
        else:
            full_hor["test"].append(task)

    return full_hor
        
full_hor = maker(data)
check_which(full_hor)
check_which(full_hor,flag = "test")


with open("Tasks/31-STR-ARC/22eb0ac0_31_mixed_str.json", 'w') as file:
    json.dump(full_hor,file,indent=4)
#"""

def transform_str_to_list(filename,_dir):
    with open (f'Tasks/{_dir}{filename}','r') as f:
        data = json.load(f)
    new_json = {"train":[],"test":[]}
    for key in data:
        for task in data[key]:
            helper = {}
            helper["input"] = str(string_to_lists(task["input"])[3:])
            helper["output"] = str(string_to_lists(task["output"])[3:])
            new_json[key].append(helper)

    filename = filename.replace("str.json","list.json")
    with open (f'Tasks/31-LIST-ARC/{filename}','w') as f:
        json.dump(new_json,f)
'''
transform_str_to_list("22eb0ac0_31_full_ver_str.json","31-STR-ARC/")
transform_str_to_list("22eb0ac0_31_full_hor_str.json","31-STR-ARC/")
'''