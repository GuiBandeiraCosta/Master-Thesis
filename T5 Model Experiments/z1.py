import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from itertools import combinations

# Define cosine similarity function
def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    norm_vector1 = torch.norm(vector1)
    norm_vector2 = torch.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)

# Load T5 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")

model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m")


# Get the vocabulary size
vocab_size = tokenizer.vocab_size

# Extract token IDs that represent numbers



numbers = "[999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999]"
numbers = numbers.replace(" ","")
inputs = tokenizer(numbers, max_length=40, padding='max_length', return_tensors="pt", truncation=True)
print(inputs)

for x in inputs["input_ids"][0]:
    print(x)
    a = tokenizer.decode(x)
    print(a)




import torch
from itertools import combinations

# Define cosine similarity function
def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    norm_vector1 = torch.norm(vector1)
    norm_vector2 = torch.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)

# Create variables for each number as torch tensors
three = torch.tensor(23)
one_two_three_four = torch.tensor(28462)
five_five_five_five = torch.tensor(27982)
exclam = torch.tensor(5)
two_zero_one_eight_zero_nine = torch.tensor(26395)
app = torch.tensor(16)
three_three_three_nine = torch.tensor(31831)
ten_thousand = torch.tensor(23899)
three_hundred_one = torch.tensor(22866)
nine_nine_nine_nine_nine_nine = torch.tensor(22215)

# Assuming model.shared returns embeddings for the given inputs
three_tensor = model.shared(three)
one_two_three_four_tensor = model.shared(one_two_three_four)
five_five_five_five_tensor = model.shared(five_five_five_five)
exclam_tensor = model.shared(exclam)
two_zero_one_eight_zero_nine_tensor = model.shared(two_zero_one_eight_zero_nine)
app_tensor = model.shared(app)
three_three_three_nine_tensor = model.shared(three_three_three_nine)
ten_thousand_tensor = model.shared(ten_thousand)
three_hundred_one_tensor = model.shared(three_hundred_one)
nine_nine_nine_nine_nine_nine_tensor = model.shared(nine_nine_nine_nine_nine_nine)

# List of tensors and their names
tensors = [
    three_tensor, one_two_three_four_tensor, five_five_five_five_tensor,
    exclam_tensor, two_zero_one_eight_zero_nine_tensor,
    app_tensor, three_three_three_nine_tensor,
    ten_thousand_tensor, three_hundred_one_tensor, nine_nine_nine_nine_nine_nine_tensor
]

tensor_names = [
    'three', 'one_two_three_four', 'five_five_five_five',
    'exclam', 'two_zero_one_eight_zero_nine',
    'app', 'three_three_three_nine',
    'ten_thousand', 'three_hundred_one', 'nine_nine_nine_nine_nine_nine'
]

# Calculate and print cosine similarity for all combinations
for i, j in combinations(range(len(tensors)), 2):
    cos_sim = cosine_similarity(tensors[i], tensors[j])
    print(f"Cosine Similarity between {tensor_names[i]} and {tensor_names[j]}: {cos_sim}")

exit()



zero_space = torch.tensor(404)

zero_no_space = torch.tensor(2468)

zero_space1 = torch.tensor(21)

zero_no_space1 = torch.tensor(29)

print(cosine_similarity(model.shared(zero_no_space),model.shared(zero_space)))
print(cosine_similarity(model.shared(zero_no_space1),model.shared(zero_space1)))
exit()
number_embeddings = {}
for token_id in range(vocab_size):
    decoded = tokenizer.decode([token_id])
    
    if decoded.isdigit():
        print(decoded)
        input_ids = torch.tensor(token_id)   
        with torch.no_grad():
            embedding = model.shared(input_ids) 
        number_embeddings[token_id] = embedding
one = 56789        
print(len(number_embeddings))

exit()



# Function to check if all pairwise cosine similarities are between -0.5 and 0.5
def all_cosine_similarities_within_range(embedding_list, min_threshold=-0.5, max_threshold=0.5):
    for i in range(len(embedding_list)):
        for j in range(i + 1, len(embedding_list)):
            cos_sim = cosine_similarity(embedding_list[i], embedding_list[j])
            if cos_sim <= min_threshold or cos_sim >= max_threshold:
                return False
    return True

# Find all valid combinations of 10 tokens
valid_combinations = []
token_ids = list(number_embeddings.keys())

for combination in combinations(token_ids, 10):
    embeddings = [number_embeddings[token_id] for token_id in combination]
    if all_cosine_similarities_within_range(embeddings):
        valid_combinations.append(combination)
        break

# Print valid combinations
print("Valid combinations of 10 different tokens with cosine similarity between -0.5 and 0.5:")
for combination in valid_combinations:
    tokens = tokenizer.convert_ids_to_tokens(combination)
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {combination}")
    print()

# Print the number of valid combinations found
print(f"Number of valid combinations found: {len(valid_combinations)}")