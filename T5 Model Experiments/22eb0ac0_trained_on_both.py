import argparse
import yaml
import os
import torch
import wandb
import json

from tqdm import tqdm # type: ignore
from torch.utils.data import DataLoader
from data_handling import load_train_data, CodeT5Dataset, load_test_data, prepare_test_data
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utilities import set_seed, calculate_accuracy,arg_list_of_ints
from nltk.metrics.distance import edit_distance

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_to_file(result,solved_examples,batch_size,f):
    helper = result
    helper["Solved_Examples"] = solved_examples
    helper["Batch"] = batch_size
    with open(f'Results/{f}_acc.json', 'a') as file:
        json.dump(helper, file)
        file.write('\n')
    

def evaluate(model, tokenizer, test_dataset, test_labels,config):
    
    model.eval()
    accuracy_sum_ver, edit_dist_sum_ver = 0, 0
    accuracy_sum_hor, edit_dist_sum_hor = 0, 0
    total_right_ver,total_right_hor = 0,0
    solved_examples = []

    with torch.no_grad():
        outputs = model.generate(test_dataset, max_new_tokens=config['max_length'])
        predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        for j in tqdm(range(len(predictions)),desc=f"Evaluation"):
            prediction = predictions[j]
            expected = test_labels[j]
            edit_distance_result = edit_distance(prediction, expected)
            accuracy_result = calculate_accuracy(prediction, expected)
            if j < 99:
                accuracy_sum_ver += accuracy_result
                edit_dist_sum_ver += edit_distance_result
                if edit_distance_result == 0:
                    total_right_ver += 1
                    solved_examples.append(j)
            else:
                accuracy_sum_hor += accuracy_result
                edit_dist_sum_hor += edit_distance_result
                if edit_distance_result == 0:
                    total_right_hor += 1
                    solved_examples.append(j)
            

    amount_of_test = len(test_labels)
    accuracy_avg_ver = round(accuracy_sum_ver / amount_of_test, 4)
    edit_dist_avg_ver = round(edit_dist_sum_ver / amount_of_test, 4)
    accuracy_avg_hor = round(accuracy_sum_hor / amount_of_test, 4)
    edit_dist_avg_hor = round(edit_dist_sum_hor / amount_of_test, 4)

    result = {"TOTAL EXACT HOR MATCHES OUT OF 100 EXAMPLES": total_right_hor,
              "TOTAL EXACT VER MATCHES OUT OF 100 EXAMPLES": total_right_ver,
              "AVERAGE EDIT DISTANCE ON 100 HOR EXAMPLES": edit_dist_avg_hor, 
              "AVERAGE ACCURACY ON 100 HOR EXAMPLES": accuracy_avg_hor,
              "AVERAGE EDIT DISTANCE ON 100 VER EXAMPLES": edit_dist_avg_ver, 
              "AVERAGE ACCURACY ON 100 VER EXAMPLES": accuracy_avg_ver,
              }
    
    return result,solved_examples, total_right_ver


def train_epoch(model, optimizer, train_dataloader, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

    for input_ids, attention_mask, labels in progress_bar:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / len(train_dataloader)
    return average_loss


def train_model(config,f,directory):
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and training data
    f = "15x15_hor_list"
    tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'])
    dataset_hor = load_train_data("15x15_hor_list",directory)
    dataset_ver = load_train_data("15x15_ver_list",directory)
    dataset = dataset_ver + dataset_hor
    
    # Load test data
    test_data_hor = load_test_data("15x15_hor_list",directory)
    test_data_ver = load_test_data("15x15_ver_list",directory)
    test_data = test_data_ver + test_data_hor
    test_input_ids, test_labels = prepare_test_data(test_data,tokenizer,device,max_length=config["max_length"])
    
    #Start Bests
    most_total_solved = 0
    best_model_state = None  
    
    # Iterate over different batch sizes
    for batch_size in config['batch_sizes']:
        set_seed(config['seed'])  
        wandb.init(project="22eb0ac0 Diff Positional Encodings", config=config,group=f, name=f"Swapped_PE_CombinedTasks_{batch_size}") 

        print(f"Starting training CombinedTasks with bs {batch_size}\n")
        train_dataloader = DataLoader(CodeT5Dataset(tokenizer, dataset, max_length=config["max_length"]), batch_size=batch_size, shuffle=True)

        model = T5ForConditionalGeneration.from_pretrained(config['checkpoint']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']))
        
        for epoch in range(1,config['num_epochs']+1):
            
            avg_loss = train_epoch(model, optimizer, train_dataloader, device, epoch)
            
            result,solved_examples,total_solved = evaluate(model, tokenizer, test_input_ids,test_labels, config)
            
            result["TRAIN LOSS"] = avg_loss
            print(result)
            wandb.log(result,step = epoch)
            
            #save_to_file(result,solved_examples,batch_size,f) #Saves a file to results folder
            
        
            if config["save_model"] == True and total_solved > most_total_solved:
                most_total_solved = total_solved
                best_model_state = model.state_dict()  # Save state dict in memory

        wandb.finish()
        
    # Save the best model at the end of all training
    if best_model_state != None:
        model.load_state_dict(best_model_state)  # Load the best model state
        model.save_pretrained(f"Model_Weights/best_model_{f}.pt")

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="Salesforce/codet5p-220m")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--batch_sizes", type= arg_list_of_ints, default=[4, 8, 16])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--tasks_dir", type=str, default="STR-ARC/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_model", type=bool, default=False)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config.update(vars(args))
    

    directory = config["tasks_dir"]
    
    for filename in os.listdir(f"Tasks/{directory}"):
        filename = filename.split(".")[0] #Removes .json
        if filename == "15x15_hor_list":
            train_model(config, filename, directory)

if __name__ == "__main__":
    main()








