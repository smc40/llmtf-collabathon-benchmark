import streamlit as st
import pandas as pd
import json
import os
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import string

load_dotenv()

if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# Create Azure OpenAI Client
# client = AzureOpenAI(
#   azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
#   api_key = os.getenv("AZURE_OPENAI_API_KEY"),
#   api_version = "2023-12-01-preview"
# )


# Create OpenAI Variables
client = OpenAI(
   api_key = os.environ.get("OPENAI_API_KEY")
)


# import data for next word prediction
with open("eval_next_word_efsa.json", 'r') as file:
    df = pd.read_json(file)



def result_json_present(filename="./data/next_word_prediction_results.json"):
    if os.path.exists(filename):
        return True
    return False
    

# get the next word prediction by submitting the prefix and then calling the openai api
def get_next_word_predictions(prefix: str, selected_model: str):
    next_word_predictions = client.chat.completions.create(
        model=selected_model, # model = "gpt-35-turbo" or "gpt-4".
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
    translator = str.maketrans('', '', string.punctuation) # translator that maps all punctuation to non
    all_predicted_next_words = set([ choice.message.content.split()[0].translate(translator) for choice in next_word_predictions.choices])

    if next_word in all_predicted_next_words:
        return 1
    else: 
        return 0


if True:
    # check if results are present, if yes, load and store it in data frame
    if result_json_present():
        if st.button("load results"):
            data = pd.read_json("./data/next_word_prediction_results.json")
            st.session_state.results_df = pd.DataFrame(data)



    # iterate over data to get all results
    models = ["gpt-4o", "gpt-35-turbo"]
    selected_model=st.radio("select model", models)
    results = []
    if st.button("run benchmark"):
        for i, row in df.head(10).iterrows():
            next_word_predictions = get_next_word_predictions(row['prefix'], selected_model)
            evaluation = evaluate_next_words(next_word_predictions, row['next-word'])
            results.append({
                'prefix': row['prefix'],
                'next-word': row['next-word'],
                'selected_model': selected_model,
                'evaluation': evaluation
            })
        st.session_state.results_df = pd.DataFrame(results)
   
   

    if st.button("save results"):
        st.session_state.results_df.to_json("./data/next_word_prediction_results.json", orient="records")



    if not st.session_state.results_df.empty: 
        eval_score = st.session_state.results_df['evaluation'].sum() / len(st.session_state.results_df)

        st.write(st.session_state.results_df)
        st.write(eval_score)

