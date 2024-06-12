import streamlit as st
import pandas as pd
import json
import os



# Streamlit application
st.title("Benchmark your LLM - check summary benchmark results")
st.subheader("The results are created using the bench mark your LLM app, created as PoC during the LLM Task Force 2024 in Bern")
st.write("")
st.write("")
st.write("")


# Loading results.json
def load_results_from_file(filename="./data/results.json"):
    if os.path.exists(filename):
        return pd.read_json(filename)
    return pd.DataFrame()

df = load_results_from_file()




# Introduce counter
if 'counter' not in st.session_state:
    st.session_state.counter = 0

def count_up():
    st.session_state.counter = (st.session_state.counter + 1) % len(df)

def count_down():
    st.session_state.counter = (st.session_state.counter - 1) % len(df)

col_1, col_2, col_3, col_4, col_5, col_6 = st.columns(6)

with col_1: 
    if st.button(label="back"):
        count_down()

with col_6: 
    if st.button(label="next"):
        count_up()




# Start of displaying results content in the app
st.header("Statistics")

col_a, col_b, col_c, col_d, col_e, col_f, col_g = st.columns(7)

with col_a:
    st.write("ROUGE-1")
    st.write(round(df.at[st.session_state.counter, 'ROUGE-1'], 3))

with col_b:
    st.write("ROUGE-L")
    st.write(round(df.at[st.session_state.counter, 'ROUGE-L'], 3))

with col_c:
    st.write("BLEU")
    st.write(round(df.at[st.session_state.counter, 'BLEU'], 3))

with col_d:
    st.write("F1")
    st.write(round(df.at[st.session_state.counter, 'F1'], 3))

with col_e:
    st.write("Recall")
    st.write(round(df.at[st.session_state.counter, 'Recall'], 3))

with col_f:
    st.write("Precision")
    st.write(round(df.at[st.session_state.counter, 'Precision'], 3))

with col_g:
    st.write("Model")
    st.write(df.at[st.session_state.counter, 'selected_model'])

st.header("Full text:")
st.write(df.at[st.session_state.counter, 'full-text'])


col3, col4= st.columns(2)

with col3:
    st.header("Gold standard")
    st.write(df.at[st.session_state.counter, 'summary'])

with col4:
    st.header("Prediction")
    st.write(df.at[st.session_state.counter, 'summary_predicted'])