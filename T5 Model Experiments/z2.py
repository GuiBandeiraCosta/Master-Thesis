import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from nltk.metrics.distance import edit_distance
from data_handling import load_test_data
from utilities import set_seed, calculate_accuracy
from tqdm import tqdm # type: ignore



def prepare_test_data(test_data,tokenizer):
    # Load and preprocess the test data
    test_data = load_test_data("22eb0ac0_31_full_ver_1closer_list", "31-Experiments/")
    test_inputs = [tokenizer(test_input, return_tensors="pt", padding='max_length', max_length=512, truncation=True)['input_ids'] for test_input, _ in test_data]
    test_inputs = test_inputs + test_inputs + test_inputs + test_inputs
    
    print(len(test_inputs))
    test_labels = [test_output for _, test_output in test_data]
    test_labels = test_labels + test_labels + test_labels + test_labels
    

    # Convert to TensorDataset and DataLoader for batching
    test_input_ids = torch.cat(test_inputs, dim=0).to('cuda')
    
    
    
    return test_input_ids, test_labels


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
model = T5ForConditionalGeneration.from_pretrained("Model_Weights/best_model_22eb0ac0_31_full_ver_1closer_list.pt").to("cuda")

test_dataset,test_labels = prepare_test_data("test_data",tokenizer)

def evaluate(model, tokenizer, test_dataset, test_labels,config):
    set_seed()
    model.eval()
    arc_original_test_acc, arg_original_test_edit = 0, 0
    accuracy_sum, edit_dist_sum = 0, 0
    total_right = 0
    solved_examples = []

    with torch.no_grad():
            
        
       
            

            
            outputs = model.generate(test_dataset, max_new_tokens=config['max_length'])
            predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            for j in tqdm(range(len(predictions)),desc=f"Evaluation"):
                
                prediction = predictions[j]
                expected = test_labels[j]
                edit_distance_result = edit_distance(prediction, expected)
                accuracy_result = calculate_accuracy(prediction, expected)
                accuracy_sum += accuracy_result
                edit_dist_sum += edit_distance_result

                if j == 0:
                    arc_original_test_acc = round(accuracy_result,4)
                    arg_original_test_edit = round(edit_distance_result,4)

                if edit_distance_result == 0:
                    total_right += 1
                    solved_examples.append(j)

    amount_of_test = len(test_labels)
    accuracy_avg = round(accuracy_sum / amount_of_test, 4)
    edit_dist_avg = round(edit_dist_sum / amount_of_test, 4)

    result = {"TOTAL EXACT MATCHES OUT OF 100 EXAMPLES": total_right,
              "ORIGINAL ARC TEST ACCURACY": arc_original_test_acc, 
              "ORIGINAL ARC TEST EDIT DISTANCE": arg_original_test_edit,
              "AVERAGE EDIT DISTANCE ON 100 EXAMPLES": edit_dist_avg, 
              "AVERAGE ACCURACY ON 100 EXAMPLES": accuracy_avg,
              }
    
    return result, solved_examples


config = {"max_length":512}
result = evaluate(model, tokenizer, test_dataset, test_labels, config)
print(result)
exit()

#print(result["TOTAL EXACT MATCHES OUT OF 100 EXAMPLES"])
#print(result)
#print(solved_examples)



def evaluate2(model, tokenizer, tokenized_test_data, config):
    test_data = load_test_data("22eb0ac0_31_full_ver_1closer_list", "31-Experiments/")
    tokenized_test_data =  [(tokenizer(test_input, return_tensors="pt", padding='max_length', max_length=config['max_length'],truncation=False).to("cuda"),test_output) for test_input,test_output in test_data] 
    predictions = []
    model.eval()
    with torch.no_grad():
        og_test_inputs, og_expected = tokenized_test_data[0]
        og_preds = model.generate(og_test_inputs['input_ids'], max_new_tokens=config["max_length"])
        og_prediction = tokenizer.decode(og_preds[0], skip_special_tokens=True)
        predictions.append(og_prediction)
        og_str_acc = round(calculate_accuracy(og_prediction, og_expected),4)
        og_edit_dist = round(edit_distance(og_prediction,og_expected),4)
        
        
        str_acc_sum, edit_dist_sum = 0,0  # Initialize the accumulator for summing the accuracies
        
        total_right = 0
        counter = 1
        achieved = []
        for test_inputs,test_output in tokenized_test_data[0:]:
            preds = model.generate(test_inputs['input_ids'], max_new_tokens=config["max_length"])
            prediction = tokenizer.decode(preds[0], skip_special_tokens=True)
            expected = test_output
            predictions.append(prediction)
            edit_distance_result = edit_distance(prediction,expected)
            str_acc_sum += calculate_accuracy(prediction, expected)
            edit_dist_sum += edit_distance_result
            if edit_distance_result == 0:
                total_right +=1
                achieved.append(counter)
            counter+=1
            
        print(achieved)
        amount_of_test = len(tokenized_test_data[0:])
        str_acc_avg = round(str_acc_sum/amount_of_test,4)
        edit_dist_avg = round(edit_dist_sum/amount_of_test,4)      
        
        print(og_str_acc, og_edit_dist,str_acc_avg,edit_dist_avg,total_right, og_prediction  )
        return predictions
preds2 = evaluate2(model,tokenizer,"a",config)

