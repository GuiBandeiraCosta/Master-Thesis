import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb # type: ignore
from nltk.metrics.distance import edit_distance
import random
import numpy as np # type: ignore
import yaml # type: ignore
from Model_Handling import Transformer
from Data_Handling import CustomTokenizer,CustomDataset,load_test_data,load_train_data
from tqdm import tqdm # type: ignore

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(config,tokenizer,batch_size,start,end):
    test_data = load_test_data(config["filename"])[start:end]

    # Create datasets and dataloaders
    test_dataset = CustomDataset(test_data, tokenizer, config["max_seq_length"])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


def test_model(transformer, tokenizer, test_loader, config, device):
    transformer.eval()

    with torch.no_grad():
        test_loader = tqdm(test_loader, desc="Evaluation", leave=False)
       
        for src_data, tgt_data in test_loader:
            src_data = src_data.to(device).squeeze(0)  # Squeeze since batch_size = 1
            tgt_data = tgt_data.to(device).squeeze(0)  # Squeeze since batch_size = 1

            # Find the first padding token in src_data to determine src_length
            pad_token = tokenizer.token_map["PAD"]
            src_length = (src_data == pad_token).nonzero(as_tuple=True)[0].min().item()
            print(src_length)
            
            pad_token = tokenizer.token_map["PAD"]
            tgt_length = (tgt_data == pad_token).nonzero(as_tuple=True)[0].min().item()
            print(tgt_length)
            # Preallocate the tgt_input tensor to max_seq_length
            tgt_input = torch.full((tgt_length,), pad_token, dtype=torch.long).to(device)
            
            # Copy src_data up to src_length into the beginning of tgt_input
            tgt_input[:src_length] = src_data[:src_length]

            # Initialize the mask to track sequences that have reached the end token
            end_mask = torch.zeros(tgt_length, dtype=torch.bool).to(device)

            for i in range(src_length, tgt_length):
                output = transformer(src_data.unsqueeze(0), tgt_input[:i].unsqueeze(0))
                next_token = output.argmax(dim=-1).squeeze(0)[-1]
            
                # Only update the position if the end token hasn't been generated
                if not end_mask[i - 1]:
                    tgt_input[i] = next_token

                # Check if the generated token is the end token
                if next_token == tokenizer.token_map["\E"]:
                    break

            predicted_tokens = tgt_input[src_length:]  # Discard the tokens from src_data
            expected_tokens = tgt_data[src_length:tgt_length]    # Align the expected tokens similarly
            
            # Calculate whether the sequence matches up to the end token
            total_right = int((predicted_tokens == expected_tokens).all().item())

            # Calculate total_edit_distance
            total_edit_distance = (predicted_tokens != expected_tokens).sum().item()

        avg_edit_distance = round(total_edit_distance, 4)

    result = {"AVG EDIT DISTANCE": avg_edit_distance, "Total Right": total_right}
    print(result)
    
    return result





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--filename", type=str, default="combined_train_test.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_model", action='store_true', default=False)
    parser.add_argument("--no_wandb", action='store_false', default=False)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config.update(vars(args))

    set_seed(42)
    
    device = torch.device(config['device'])
    print(device)

    tokenizer = CustomTokenizer()
    
    transformer = Transformer(config["src_vocab_size"], config["tgt_vocab_size"], config['d_model'], config['num_heads'], 
                              config['num_layers'], config['d_ff'], config['max_seq_length'] , config['dropout']).to(device)
    
    weight_path = "combined_train_test.json_model_weights_40.pth"
    transformer.load_state_dict(torch.load(weight_path, map_location=device))
    

    test_loader = load_data(config,tokenizer,batch_size=1,start=0,end=100000)

    test_model(transformer,tokenizer,test_loader,config,device)
    
if __name__ == "__main__":
    main()