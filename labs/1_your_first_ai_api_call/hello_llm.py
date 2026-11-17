#!/usr/bin/env python

import os
from openai import OpenAI

API_KEY:str = os.getenv("OPENAI_API_KEY")
API_URL:str = os.getenv("OPENAI_API_BASE")

prompt:str = "Write a one-sentence story about a robot."

client = OpenAI(api_key=API_KEY, base_url=API_URL)


print("\ttemp=0.0\t|\ttemp=2.0")
print("-"*50)

for i in range(10): 
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
        {"role": "user", "content": prompt}],
        extra_body={
            "cache": {
                "no-cache": True
            }
        },
        temperature=0.0
    )

    print(response.choices[0].message.content[:25], end=" | ")

    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
        {"role": "user", "content": prompt}],
        extra_body={
            "cache": {
                "no-cache": True
            }
        },
        temperature=2.0
    )

    print(response.choices[0].message.content[:25])
