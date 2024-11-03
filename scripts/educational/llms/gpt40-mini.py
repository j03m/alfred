import os
from openai import OpenAI
import trafilatura
import json

with open("keys/openai.txt", 'r') as file:
    api_key = file.readline().strip()


# Set an environment variable
os.environ['OPENAI_API_KEY'] = api_key

ticker = "AAPL"

# Load and extract main content from the HTML file
with open("/Users/jmordetsky/alfred/news/ADP/20241102/438ac011.txt", 'r', encoding='utf-8') as file:
    html_content = file.read()
    contents = trafilatura.extract(html_content)

# Check if content was extracted
if not contents:
    raise ValueError("No content could be extracted from the HTML.")

# Construct the prompt
prompt = (
    f"The following is a news article about {ticker} stock. Please create a JSON response with only 3 fields:\n"
    f"- relevance: from 0 to 1, where 1 means the article is 100% about {ticker} and 0 means it's not about {ticker} at all.\n"
    f"- sentiment: from -1 to 1, where 1 is overwhelmingly positive about {ticker} and -1 is overwhelmingly negative about {ticker}.\n"
    f"- outlook: from -1 to 1, where 1 is a BULLISH outlook on {ticker} and -1 is a BEARISH outlook on {ticker}."
    f"Here is the article:\n{contents}\n\n"
    f"JSON Response:"
)

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a stock research assistant."},
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]
)

json_string = completion.choices[0].message.content
json_string_cleaned = json_string.strip('```json\n').strip('```').strip()
parsed_response = json.loads(json_string_cleaned)

# Access and print the parsed data
print('Relevance:', parsed_response['relevance'])
print('Sentiment:', parsed_response['sentiment'])
print('Outlook:', parsed_response['outlook'])