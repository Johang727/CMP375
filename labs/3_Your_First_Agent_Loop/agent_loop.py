import os
from openai import OpenAI

OPENAI_API_KEY:str = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE:str = os.getenv("OPENAI_API_BASE")

messages:list[dict] = [
    {"role": "system", "content": "You are a helpful assistant."}
    ]

questions:list[str] = [
    "What is an agent?", 
    "How is that different from a chatbot?",
    "Give me one example."
]

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

for question in questions:
    messages.append({
        "role":"user",
        "content":question
    })
    while True:
        response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=messages,
        )

        assistant_message:str = response.choices[0].message
        finish_reason:str = response.choices[0].finish_reason

        if finish_reason == "stop":
            print(question)
            print(assistant_message.content)
            messages.append(assistant_message)
            break
        else: 
            break