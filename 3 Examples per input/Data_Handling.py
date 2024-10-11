import torch
import json
from torch.utils.data import Dataset
class CustomTokenizer:
    def __init__(self):
        self.token_map = {
            "0": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
            "5": 7,
            "6": 8,
            "7": 9,
            "8": 10,
            "9": 11,
            "[[": 12,
            "[": 13,
            "]": 14,
            "]]": 15,
            "INa": 16,
            "INb": 17,
            "INc": 18,
            "INd": 19,
            "OUTa": 20,
            "OUTb": 21,
            "OUTc": 22,
            "OUTd": 23,
            "\S": 1,
            "\E": 25,
            "PAD": 24
        }
        self.inv_token_map = {v: k for k, v in self.token_map.items()}

    def tokenize(self, text):
        tokens = ["\S"]
        i = 0
        while i < len(text):
            if text[i:i+2] == "[[":
                tokens.append("[[")
                i += 2
            elif text[i:i+2] == "]]":
                tokens.append("]]")
                i += 2
            elif text[i] in self.token_map:
                tokens.append(text[i])
                i += 1
            else:
                i += 1  # Skip any character not in the token map
        tokens.append("\E")
        return [self.token_map[token] for token in tokens]

    def detokenize(self, token_ids):
        return ''.join([self.inv_token_map[token_id] for token_id in token_ids if token_id in self.inv_token_map])

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_tokens = self.tokenizer.tokenize(src)
        tgt_tokens = self.tokenizer.tokenize(tgt)
        src_padded = src_tokens + [self.tokenizer.token_map["PAD"]] * (self.max_seq_length - len(src_tokens))
        tgt_padded = tgt_tokens + [self.tokenizer.token_map["PAD"]] * (self.max_seq_length - len(tgt_tokens))
        return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)

def load_train_data(filepath):
    with open(filepath,"r") as file:
        data = json.load(file)
    return [(task["input"], task["output"]) for task in data["train"]]

def load_test_data(filepath):
    with open(filepath,"r") as file:
        data = json.load(file)
    return [(task["input"], task["output"]) for task in data["test"]]
