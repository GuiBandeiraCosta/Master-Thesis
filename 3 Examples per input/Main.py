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

def load_data(config,tokenizer):
    # Load data
    train_data = load_train_data(config["filename"])
    test_data = load_test_data(config["filename"])

    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_data, tokenizer, config["max_seq_length"])
    test_dataset = CustomDataset(test_data, tokenizer, config["max_seq_length"])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader,test_loader

def train_model(transformer,optimizer,criterion,train_loader,config,device,epoch):
    total_loss = 0
    transformer.train()
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    accumulation_steps = 64  # Simulate batch size of 64
    optimizer.zero_grad()

    for i, (src_data, tgt_data) in enumerate(train_loader):
        src_data, tgt_data = src_data.to(device), tgt_data.to(device)
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, config["tgt_vocab_size"]), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        train_loader.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / len(train_loader)
    
    
    
    # for src_data, tgt_data in train_loader:
    #     src_data, tgt_data = src_data.to(device), tgt_data.to(device)
    #     optimizer.zero_grad()
    #     output = transformer(src_data, tgt_data[:, :-1]) #Teacher Forcing
    #     loss = criterion(output.contiguous().view(-1, config["tgt_vocab_size"]), tgt_data[:, 1:].contiguous().view(-1))
    #     loss.backward()
    #     total_loss += loss.item()
    #     optimizer.step()
    #     train_loader.set_postfix(loss=f"{loss.item():.4f}")
    
    # average_loss = total_loss / len(train_loader)    
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    return average_loss





def test_model(transformer, tokenizer, test_loader, config, device, epoch):
    size_of_testset = len(test_loader.dataset)
    total_edit_distance = 0
    total_right = 0
    transformer.eval()

    with torch.no_grad():
        #test_loader = tqdm(test_loader, desc=f"Evaluation Epoch {epoch}", leave=False)
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
                print(i)
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
            
            print(expected_tokens)
            print(predicted_tokens)
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




def run(transformer,tokenizer,train_loader,test_loader,config,device):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_map["PAD"]).to(device)
    optimizer = optim.AdamW(transformer.parameters(), lr=0.0001)
    # Training loop
    for epoch in range(1,config["num_epochs"]):
    
        avg_loss = train_model(transformer,optimizer,criterion,train_loader,config,device,epoch)
        
        # if epoch %2 == 0:
        #     result = test_model(transformer,tokenizer,test_loader,config,device,epoch)
        #     result["TRAIN LOSS"] = avg_loss

        #     if config['no_wandb']:
        #         wandb.log(result,step = epoch)
            
        if epoch %10 == 0:
            if config["save_model"]:
                torch.save(transformer.state_dict(), f"{config['filename']}_model_weights_{epoch}.pth")
                   






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--filename", type=str, default="combined_train_test.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=41)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_model", action='store_true', default=False)
    parser.add_argument("--no_wandb", action='store_false', default=True)
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
    
    train_loader, test_loader = load_data(config,tokenizer)
    
    if config['no_wandb']:
        wandb.init(project="Several ARC Problems",config=config, name=f"{config['filename']}")

    run(transformer,tokenizer,train_loader,test_loader,config,device)

    if config['no_wandb']:
        wandb.finish() 

if __name__ == "__main__":
    main()