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
    size_of_testset = len(test_loader.dataset)
    total_edit_distance = 0
    total_right = 0
    transformer.eval()

    with torch.no_grad():
        for src_data, tgt_data in test_loader:
            src_data = src_data.to(device)
            tgt_data = tgt_data.to(device)
            test_batch_size = src_data.size(0)
            
            # Preallocate the tgt_input tensor to max_seq_length
            tgt_input = torch.full((test_batch_size, config["max_seq_length"]), tokenizer.token_map["PAD"], dtype=torch.long).to(device)
            tgt_input[:, 0] = tokenizer.token_map["\S"]  # Set the start token at the first position

            # Initialize the mask to track sequences that have reached the end token
            end_mask = torch.zeros(test_batch_size, dtype=torch.bool).to(device)
    
            for i in range(1, config["max_seq_length"]):
                output = transformer(src_data, tgt_input[:, :i])
                next_token = output.argmax(dim=-1)[:, -1]

                # Only update sequences that have not yet generated the end token
                tgt_input[:, i] = torch.where(end_mask, tgt_input[:, i], next_token)
                
                # Update the mask to include sequences that have just generated the end token
                end_mask |= (next_token == tokenizer.token_map["\E"])

                # If all sequences have generated the end token, stop early
                if end_mask.all():
                    break

            predicted_tokens = tgt_input[:, 1:]  # Discard the initial start token
            expected_tokens = tgt_data[:, 1:]    # Align the expected tokens similarly

            # Calculate total_right: sequences must match up to the end token
            correct_sequences = (predicted_tokens == expected_tokens).all(dim=1)
            total_right += correct_sequences.sum().item()

            # Calculate total_edit_distance for the entire batch
            mismatches = (predicted_tokens != expected_tokens)
            total_edit_distance += mismatches.sum().item()

        avg_edit_distance = round(total_edit_distance / size_of_testset, 4)

    result = {"AVG EDIT DISTANCE": avg_edit_distance, "Total Right": total_right}
    print(result)
    
    return result




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--filename", type=str, default="22eb0ac0_16x16_testing.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda:1")
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
    
    weight_path = "22eb0ac0_max_15x15_Nocommas_100K_dp=0.1_model_weights_12.pth"
    transformer.load_state_dict(torch.load(weight_path, map_location=device))
    

    test_loader = load_data(config,tokenizer,batch_size=128,start=0,end=1000)

    test_model(transformer,tokenizer,test_loader,config,device)
    
if __name__ == "__main__":
    main()