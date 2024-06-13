import streamlit as st
import pandas as pd
import json
import os
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI Variables
# client = AzureOpenAI(
#   azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
#   api_key = os.getenv("AZURE_OPENAI_API_KEY"),
#   api_version = "2023-12-01-preview"
# )
client = OpenAI(
   api_key = os.environ.get("OPENAI_API_KEY")
)


# import data for next word prediction
with open("eval_next_word_efsa.json", 'r') as file:
    df = pd.read_json(file)


# get the next word prediction by submitting the prefix and then calling the openai api
def get_next_word_predictions(prefix: str):
    next_word_predictions = client.chat.completions.create(
        model="gpt-4o", # model = "gpt-35-turbo" or "gpt-4".
        max_tokens=5,
        temperature=1,   
        n=10,
        messages=[
            {"role": "system", "content": "just continue the following sentences"},
            {"role": "user", "content": prefix},
        ]
    )
    return next_word_predictions

# compare the predicted next words to the golden next word by submittng predictions and gold standard
def evaluate_next_words(next_word_predictions, next_word: str):
    all_predicted_next_words = set([ choice.message.content.split()[0] for choice in next_word_predictions.choices])
    st.write(all_predicted_next_words)
    if next_word in all_predicted_next_words:
        return 1
    else: 
        return 0

i=1

st.write(df.at[i, 'prefix'])

next_word_predictions = get_next_word_predictions(df.at[i, 'prefix'])

st.write(next_word_predictions.choices)

evalutation = evaluate_next_words(next_word_predictions, df.at[i, 'next-word'])

st.write(evalutation)

# for choice in response.choices:
#  print(choice.message.content)


# evaluate("Swissmedic is the","greatest")