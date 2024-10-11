import random
import numpy as np # type: ignore
import torch

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def arg_list_of_ints(arg):
    return [int(x) for x in arg.split(',')]

def calculate_accuracy(preds, expected):
    total = sum(x == y for x, y in zip(preds, expected))
    return min(total / len(expected), total / len(preds))

def convert_lists_to_string(matrix):
    return ' '.join(''.join(map(str, sublist)) for sublist in matrix)

def string_to_lists(input_string):
    # Split the string by spaces to get individual segments
    segments = input_string.split(" ")
    # Convert each segment into a list of integers
    matrix = [[int(char) for char in segment] for segment in segments]
    return matrix

def dict_to_lists(og_dict):
    new_dict = {}
    input = og_dict["input"]
    output = og_dict["output"]
    new_dict["input"] = string_to_lists(input)
    new_dict["output"] = string_to_lists(output)
    return new_dict

