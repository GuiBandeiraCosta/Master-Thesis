from Data_Handling import CustomTokenizer
import json
from transformers import AutoTokenizer
counter = 0

with open("combined_train_test.json","r") as f:
    data = json.load(f)
    
a = CustomTokenizer()
a = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
for key in data:
    for x in data[key]:
        sss = a.tokenize(x["output"])
        if len(sss) > counter:
            counter = len(sss)

print(counter)