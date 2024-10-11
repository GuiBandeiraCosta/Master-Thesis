import json
from torch.utils.data import Dataset
from torch import cat as torch_cat
import os


def fix_json_file(f):

    json_list = []
    with open(f'Results/{f}_acc.json', 'r') as file:
        for line in file:

            stripped_line = line.strip()
            if stripped_line:

                json_object = json.loads(stripped_line)
                json_list.append(json_object)

    # Write the list of JSON objects to a new file as a JSON array
    with open(f'Results/{f}_acc.json', 'w') as outfile:
        outfile.write("[\n")

        for i, obj in enumerate(json_list):
            json_string = json.dumps(obj, separators=(',', ':'))
            outfile.write(json_string)
            if i < len(json_list) - 1:
                outfile.write(",\n")

        outfile.write("\n]")

def save_to_file(result,solved_examples,batch_size,f):
    helper = result
    helper["Solved_Examples"] = solved_examples
    helper["Batch"] = batch_size
    with open(f'Results/{f}_acc.json', 'a') as file:
        json.dump(helper, file)
        file.write('\n')


def load_train_data(f,directory):
    with open(f'Tasks/{directory}{f}.json', 'r') as file:
        data = json.load(file)
    return [(str(task["input"]), str(task["output"])) for task in data["train"]]

def load_test_data(f,directory):
    with open(f'Tasks/{directory}{f}.json', 'r') as file:
        data = json.load(file)
    
    return [(str(task["input"]), str(task["output"])) for task in data["test"]]



def load_train_data_from_dir(directory):
    total_train_data =  []
    for filename in os.listdir(f"Tasks/{directory}"):
        filename = filename.split(".")[0] #Removes .json
        total_train_data += load_train_data(filename,directory)
    return total_train_data

def load_test_data_from_dir(directory):
    total_test_data = {}
    for filename in os.listdir(f"Tasks/{directory}"):
        filename = filename.split(".")[0] #Removes .json
        total_test_data[filename] = load_test_data(filename,directory)
    return total_test_data


def prepare_test_data(test_data,tokenizer,device,max_length=512):
    # Load and preprocess the test data
    test_inputs = [tokenizer(test_input, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)['input_ids'] for test_input, _ in test_data]
    test_labels = [test_output for _, test_output in test_data]
    
    test_input_ids = torch_cat(test_inputs, dim=0).to(device)
    return test_input_ids, test_labels


class CodeT5Dataset(Dataset):
    def __init__(self, tokenizer, data, max_length=160):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        inputs = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', return_tensors="pt", truncation=True)
        targets = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', return_tensors="pt", truncation=True)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        target_ids = targets['input_ids'].squeeze()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100 # For Cross Entropy Loss
        return input_ids,attention_mask, target_ids
