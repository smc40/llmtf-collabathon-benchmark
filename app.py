import streamlit as st
import pandas as pd
import json
import random
from time import time
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

import spacy

nlp = spacy.load("en_core_web_sm")

# # Download NLTK data
# nltk.download("punkt")

# Set the mode (either "dummy" or "real")
MODE = "dummy"


st.set_page_config(layout="wide")


# Define a function to validate the JSON structure
def validate_json_structure(data):
    if not isinstance(data, list):
        return False
    for entry in data:
        if not isinstance(entry, dict):
            return False
        if set(entry.keys()) != {"full-text", "summary", "source", "URL"}:
            return False
    return True


# Function to fake a summary prediction in dummy mode
def dummy_summary_prediction(summary):
    words = summary.split()
    random.shuffle(words)
    return " ".join(words)


# Function to calculate ROUGE and BLEU scores
def calculate_scores(reference, prediction):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, prediction)
    rouge1 = rouge_scores["rouge1"].fmeasure
    rougeL = rouge_scores["rougeL"].fmeasure
    bleu = sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=SmoothingFunction().method1,
    )
    return rouge1, rougeL, bleu


@st.cache_data
def process_uploaded_file(uploaded_file):
    data = json.load(uploaded_file)
    if validate_json_structure(data):
        df = pd.DataFrame(data)
        return df
    else:
        return None


# Streamlit application
st.title("Benchmark your LLM on creating a Summary")
st.write("PoC build during the Collabathon of the LLM Taskforce 2024 in Bern")

st.text_area(
    "specify the prompt you want to use (if not provided in the JSON)",
    value="Summarize this document to a max of 2000 words",
)


st.write(
    "Upload a JSON file containing documents with full-text, summary, source, and URL. The tool will benchmark the performance of your LLM in generating summaries."
)

uploaded_file = st.file_uploader("Upload JSON File", type="json")

import nlp

if uploaded_file:
    df = process_uploaded_file(uploaded_file)
    if df is not None:

        # Insights per source
        insights = df["source"].value_counts().to_frame().reset_index()
        insights["tokens_len_summary"] = len(nlp(df["summary"]))

        insights.columns = ["Source", "Count"]
        insights["Flag"] = insights["Source"].map(
            {"MHRA": "🇬🇧", "EFSA": "🇪🇺", "SMC": "🇨🇭"}
        )
        st.write("Insights per source:")
        st.write(insights)

        if st.button("Run Benchmark"):
            start_time = time()
            summary_predicted = []
            rouge1_scores = []
            rougeL_scores = []
            bleu_scores = []

            for index, row in df.iterrows():
                if MODE == "dummy":
                    predicted = dummy_summary_prediction(row["summary"])
                else:
                    # Real mode: call OpenAI API (assuming there's a function for that)
                    predicted = openai_summary_prediction(row["full-text"])

                summary_predicted.append(predicted)

                # Calculate scores for each row
                rouge1, rougeL, bleu = calculate_scores(row["summary"], predicted)
                rouge1_scores.append(rouge1)
                rougeL_scores.append(rougeL)
                bleu_scores.append(bleu)

            df["summary_predicted"] = summary_predicted
            df["ROUGE-1"] = rouge1_scores
            df["ROUGE-L"] = rougeL_scores
            df["BLEU"] = bleu_scores

            end_time = time()
            total_time = end_time - start_time

            # Calculate average scores per source
            scores = []
            for source in df["source"].unique():
                source_data = df[df["source"] == source]
                avg_rouge1 = source_data["ROUGE-1"].mean()
                avg_rougeL = source_data["ROUGE-L"].mean()
                avg_bleu = source_data["BLEU"].mean()
                scores.append(
                    {
                        "Source": source,
                        "ROUGE-1": avg_rouge1,
                        "ROUGE-L": avg_rougeL,
                        "BLEU": avg_bleu,
                        "Time (s)": total_time,
                    }
                )

            scores_df = pd.DataFrame(scores)
            st.write("LLM predictions and scores per row:")
            st.write(df)

            st.write("Measures:")
            st.write(scores_df)

            with st.expander("How does ROUGE score work?"):
                st.write(
                    "ROUGE score measures the similarity between the machine-generated summary and the reference summaries using overlapping n-grams, word sequences that appear in both the machine-generated summary and the reference summaries. The most common n-grams used are unigrams, bigrams, and trigrams. ROUGE score calculates the recall of n-grams in the machine-generated summary by comparing them to the reference summaries. ROUGE-N (N-gram) scoring. ROUGE-L (Longest Common Subsequence) scoring"
                )
                st.write(
                    "https://medium.com/@sthanikamsanthosh1994/understanding-bleu-and-rouge-score-for-nlp-evaluation-1ab334ecadcb#:~:text=ROUGE%20score%20measures%20the%20similarity,unigrams%2C%20bigrams%2C%20and%20trigrams."
                )

            with st.expander("How does BLEU score work?"):
                st.write(
                    "BLEU score measures the similarity between the machine-translated text and the reference translations using n-grams, which are contiguous sequences of n words. The most common n-grams used are unigrams (single words), bigrams (two-word sequences), trigrams (three-word sequences), and so on."
                )
                st.write(
                    "https://medium.com/@sthanikamsanthosh1994/understanding-bleu-and-rouge-score-for-nlp-evaluation-1ab334ecadcb#:~:text=ROUGE%20score%20measures%20the%20similarity,unigrams%2C%20bigrams%2C%20and%20trigrams."
                )
    else:
        st.error("The JSON file does not match the required structure.")
