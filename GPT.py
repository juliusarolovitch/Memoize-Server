#GPT-4o API calls
# implement GPT 4o such that it has memory etc
# integrate it into API
from openai import OpenAI

class Text:
    def __init__(self, text, apikey):
        self.client = OpenAI(api_key = apikey)
        self.text = text

    def to_gpt(self):
        completion = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": self.text}
        ]
        )
        return completion.choices[0].message.content

