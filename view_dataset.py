from datasets import load_dataset
dataset = load_dataset("RaDeR/final_complete_Math_NuminaMath_allquerytypes")

# If the dataset is private, you will need to authenticate
# You can authenticate with Hugging Face Hub using your token
from huggingface_hub import login
login(token = "abc")  # This will prompt you to log in with your Hugging Face token

# Display the dataset
print(dataset['train'][6])
print(len(dataset['train']))


