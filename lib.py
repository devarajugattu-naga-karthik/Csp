# Importing necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Loading the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can also try "gpt2-medium", "gpt2-large" or "gpt2-xl" for more powerful models
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate text based on a user prompt
def generate_text(prompt, max_length=200):
    # Encoding the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generating text from the model
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7, top_p=0.9, top_k=50)
    
    # Decoding the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# Test the text generation model
user_prompt = "Artificial intelligence is revolutionizing the world"
generated_paragraph = generate_text(user_prompt)
print(generated_paragraph)
