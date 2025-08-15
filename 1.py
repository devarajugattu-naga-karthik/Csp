import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=100):
    """Generates text using a pre-trained GPT-2 model."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "Artificial intelligence is transforming the world by"
generated_text = generate_text(prompt)
print(generated_text)
