import streamlit as st
import pandas as pd
import json
import random
from time import time
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

import os
from openai import AzureOpenAI

import spacy

nlp = spacy.load("en_core_web_sm")

# # Download NLTK data
# nltk.download("punkt")

# Set the mode (either "dummy" or "real")
MODE = "dummy"
models = ["gpt-4", "gpt-35-turbo"]


st.set_page_config(layout="wide")


@st.cache_data
def generate_summary(
    summary: str,
    selected_models: str,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
    )

    response = client.chat.completions.create(
        model=selected_models,  # model = "gpt-35-turbo" or "gpt-4".
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


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

default_user_prompt = st.text_area(
    "Specify the prompt you want to use (if not provided in the JSON)",
    value="Summarize this document to a max of 2000 words",
)

selected_model = st.radio("Select model to benchmark 📈", models)

st.write(
    "Upload a JSON file containing documents with full-text, summary, source, and URL. The tool will benchmark the performance of your LLM in generating summaries."
)


# Function to calculate token length
def calculate_token_length(text):
    doc = nlp(text)
    return len(doc)


def calculate_f1_recall_precision(reference, predicted):
    # Tokenize the reference and predicted summaries
    ref_tokens = set(reference.split())
    pred_tokens = set(predicted.split())

    # Calculate precision, recall, and F1 score
    precision = (
        len(ref_tokens & pred_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    )
    recall = (
        len(ref_tokens & pred_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return f1, recall, precision


uploaded_file = st.file_uploader("Upload JSON File", type="json")

if uploaded_file:
    df = process_uploaded_file(uploaded_file)
    if df is not None:

        # Insights per source
        insights = df["source"].value_counts().to_frame().reset_index()
        insights.columns = ["Source", "Count"]
        insights["source_flags"] = insights["Source"].map(
            {"MHRA": "🇬🇧", "EFSA": "🇪🇺", "SMC": "🇨🇭"}
        )

        # Group by the 'source' column
        grouped_df = df.groupby("source")

        # Calculate token length for each summary
        df["tokens_len_full_text"] = df["full-text"].apply(calculate_token_length)
        insights["tokens_len_full_text"] = (
            grouped_df["tokens_len_full_text"].mean().values
        )

        # Calculate token length for each summary
        df["tokens_len_summary"] = df["summary"].apply(calculate_token_length)
        insights["tokens_len_summary"] = grouped_df["tokens_len_summary"].mean().values

        st.write("Insights per source:")
        st.write(insights)

        if st.button("Run Benchmark"):
            start_time = time()
            summary_predicted = []
            rouge1_scores = []
            rougeL_scores = []
            bleu_scores = []
            f1_scores = []
            recall_scores = []
            precision_scores = []

            for index, row in df.iterrows():
                if MODE == "dummy":
                    predicted = dummy_summary_prediction(row["summary"])
                else:
                    # Real mode: call OpenAI API (assuming there's a function for that)
                    prompt = row.get("prompt", default_user_prompt)
                    predicted = generate_summary(
                        row["full-text"], selected_model, prompt
                    )

                summary_predicted.append(predicted)

                # Calculate scores for each row
                rouge1, rougeL, bleu = calculate_scores(row["summary"], predicted)
                f1, recall, precision = calculate_f1_recall_precision(
                    row["summary"], predicted
                )

                rouge1_scores.append(rouge1)
                rougeL_scores.append(rougeL)
                bleu_scores.append(bleu)
                f1_scores.append(f1)
                recall_scores.append(recall)
                precision_scores.append(precision)

            df["summary_predicted"] = summary_predicted

            # Calculate token length for each summary
            df["tokens_len_summary_predicted"] = df["summary"].apply(
                calculate_token_length
            )
            insights["tokens_len_summary_predicted"] = (
                grouped_df["tokens_len_summary_predicted"].mean().values
            )

            df["ROUGE-1"] = rouge1_scores
            df["ROUGE-L"] = rougeL_scores
            df["BLEU"] = bleu_scores
            df["F1"] = f1_scores
            df["Recall"] = recall_scores
            df["Precision"] = precision_scores

            end_time = time()
            total_time = end_time - start_time

            # Calculate average scores per source
            scores = []
            for source in df["source"].unique():
                source_data = df[df["source"] == source]
                avg_rouge1 = source_data["ROUGE-1"].mean()
                avg_rougeL = source_data["ROUGE-L"].mean()
                avg_bleu = source_data["BLEU"].mean()
                avg_recall = source_data["Recall"].mean()
                avg_precision = source_data["Precision"].mean()
                avg_f1 = source_data["F1"].mean()
                scores.append(
                    {
                        "Model": selected_model,
                        "Source": source,
                        "ROUGE-1": avg_rouge1,
                        "ROUGE-L": avg_rougeL,
                        "BLEU": avg_bleu,
                        "Recall": avg_recall,
                        "Precision": avg_precision,
                        "F1": avg_f1,
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
