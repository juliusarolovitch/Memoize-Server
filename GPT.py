#GPT-4o API calls
# implement GPT 4o such that it has memory etc
# integrate it into API
from openai import OpenAI

class Text:
    def __init__(self, text, apikey, llm):
        self.client = OpenAI(api_key = apikey)
        self.text = text
        self.llm = llm
    def to_gpt(self):
        completion = self.client.chat.completions.create(
        model=self.llm,
        messages=[
            {"role": "system", "content": "You are the clone / digital twin of the user. You learn everything about them and act like them whenever you are in a conversation. Speak in first person view."},
            {"role": "user", "content": self.text}
        ]
        )
        return completion.choices[0].message.content
text = Text("Wny do you finish your homework only so late just before the deadline?", "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")
text.to_gpt()