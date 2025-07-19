import os
import dotenv as env
from openai import OpenAI

env.load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

num_days = 7
num_childs = 0
activity = "Praia"

prompt = f"Crie um roteiro de viagem de {num_days} dias, para uma familia com {num_childs} crianças, que gostam de {activity}"

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistent and your goal is help other people to planning their travels. Your planning needs to have maximum of 100 words",
        },
        {"role": "user", "content": prompt},
    ],
)

roteiro_viagem = response.choices[0].message.content
print(roteiro_viagem)
