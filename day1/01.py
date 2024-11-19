import config
import os
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": 
         """
         Translate the given word into Korean.
         Apple: 사과,
         Banana: 바나나,
         Castle: 성,
         Dawn: 새벽,
         Event: 행사
         """
         },
        {
            "role": "user",
            "content": "Dog: "
        }
    ]
)

print(completion.choices[0].message)