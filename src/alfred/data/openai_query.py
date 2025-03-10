import os
from openai import OpenAI
import trafilatura
import json


class OpenAiQuery:
    def __init__(self, key_file="keys/openai.txt", model="gpt-4o-mini"):
        with open(key_file, 'r') as file:
            api_key = file.readline().strip()

        # Set an environment variable
        # os.environ['OPENAI_API_KEY'] = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def query(self, prompt, role):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": role},
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ]
        )
        return completion

    def news_query(self, body, ticker):

        # Load and extract main content from the HTML file
        contents = trafilatura.extract(body)

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

        completion = self.query(prompt, f"you are a helpful stock analyst researching {ticker}")

        json_string = completion.choices[0].message.content
        json_string_cleaned = json_string.strip('```json\n').strip('```').strip()
        parsed_response = json.loads(json_string_cleaned)
        return parsed_response