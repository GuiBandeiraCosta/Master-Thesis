
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
hello = []
for a in range(20):
    helper = []
    for b in range(20):
        helper.append(b)
    hello.append(helper)
    

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
# Tokenize the text

inputs = tokenizer("[[ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9] [ 2 6 4 2 2 3 3 9 5 2 2 7 4 7 9]]", max_length=467,return_tensors="pt", truncation=True)

# Print original token ids shape
print("Original shape:", inputs.input_ids.shape)
print(inputs)
exit()
for x in inputs["input_ids"][0]:
    print(tokenizer.decode(x))

# Find the token ID for the comma
comma_token_id = tokenizer.convert_tokens_to_ids(',')

# Create a mask that keeps all tokens except the comma
mask = inputs.input_ids != comma_token_id

# Apply the mask to remove commas and keep the format
inputs["input_ids"] = inputs.input_ids[mask].view(1, -1)
inputs["attention_mask"] = inputs.attention_mask[mask].view(1, -1)


# Print the results
print("Filtered input IDs shape:")
print("Filtered attention mask shape:", inputs["input_ids"].shape)
print(inputs["attention_mask"].shape)
#print("Decoded inputs without commas:", decoded_inputs)