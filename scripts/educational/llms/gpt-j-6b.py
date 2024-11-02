import torch
from transformers import GPTJForCausalLM, AutoTokenizer
import trafilatura
from alfred.devices import set_device

device = set_device()
model_name = 'EleutherAI/gpt-j-6B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(
    model_name,
    revision='float16',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)


model.to(device)

# Load and extract main content from the HTML file
with open("news/AAPL/20241029/b28cd127.txt", 'r', encoding='utf-8') as file:
    html_content = file.read()
    contents = trafilatura.extract(html_content)

# Check if content was extracted
if not contents:
    raise ValueError("No content could be extracted from the HTML.")

# Construct the prompt
prompt = (
    f"The following is a news article about AAPL stock. Please create a JSON response with only 3 fields:\n"
    f"- relevance: from 0 to 1, where 1 means the article is 100% about AAPL and 0 means it's not about AAPL at all.\n"
    f"- sentiment: from -1 to 1, where 1 is overwhelmingly positive about AAPL and -1 is overwhelmingly negative about AAPL.\n"
    f"- outlook: from -1 to 1, where 1 is a BULLISH outlook on AAPL and -1 is a BEARISH outlook on AAPL.\n"
    f"Here is the article:\n{contents}\n\n"
    f"JSON Response:"
)

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_length = inputs['input_ids'].shape[1]

# Model's maximum context length
model_max_length = model.config.max_position_embeddings  # 2048 for GPT-J-6B

# Desired number of tokens to generate
desired_output_length = 256  # Adjust based on expected output length

# Check if total length exceeds model's maximum context length
total_length = input_length + desired_output_length
if total_length > model_max_length:
    # Truncate input to fit within context window
    tokens_to_remove = total_length - model_max_length
    print(f"Input prompt is too long ({input_length} tokens). Truncating by {tokens_to_remove} tokens.")
    inputs['input_ids'] = inputs['input_ids'][:, tokens_to_remove:]
    input_length = inputs['input_ids'].shape[1]
    total_length = input_length + desired_output_length

# Set max_length for generation
max_length = total_length

# Generate text
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Extract only the generated part after "JSON Response:"
split_text = generated_text.split("JSON Response:")
if len(split_text) > 1:
    generated_part = split_text[1].strip()
else:
    generated_part = generated_text.strip()

print("Generated JSON Response:")
print(generated_part)
