import streamlit as st
import pandas as pd
import json
import random
import os
from time import time
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import AzureOpenAI
import time as times

# Set the mode (either "dummy" or "real")
MODE = "real"
models = ["gpt-4", "gpt-35-turbo"]

st.set_page_config(layout="wide")

# Streamlit application
st.title("Benchmark your LLM on creating a Summary")
st.write("PoC build during the Collabathon of the LLM Taskforce 2024 in Bern")


# Create data directory if it doesn't exist
if not os.path.exists("./data"):
    os.makedirs("./data")

df = pd.DataFrame()


def exitster(filename="./data/results.json"):
    if os.path.exists(filename):
        return True
    return False


show = exitster()

# Streamlit application
st.title("Benchmark your LLM on creating a Summary")
st.write("PoC build during the Collabathon of the LLM Taskforce 2024 in Bern")


# Create data directory if it doesn't exist
if not os.path.exists("./data"):
    os.makedirs("./data")

df = pd.DataFrame()


def exitster(filename="./data/results.json"):
    if os.path.exists(filename):
        return True
    return False


show = exitster()


@st.cache_data
def generate_summary(
    full_text: str,
    selected_models: str,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
    )

    msg = [
        {
            "role": "system",
            "content": f"""{system_prompt}""",
        },
        {
            "role": "user",
            "content": f"""\ntext to summarize: {full_text} \n {user_prompt}""",
        },
    ]
    print(msg)

    response = client.chat.completions.create(
        model=selected_models,  # model = "gpt-35-turbo" or "gpt-4".
        messages=msg,
    )

    return response.choices[0].message.content


@st.cache_data
def validate_json_structure(data):
    if not isinstance(data, list):
        return False
    for entry in data:
        if not isinstance(entry, dict):
            return False
        if set(entry.keys()) != {"full-text", "summary", "source", "URL"}:
            return False
    return True


def dummy_summary_prediction(summary):
    words = summary.split()
    random.shuffle(words)
    return " ".join(words)


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
    df = pd.DataFrame(data)
    return df
    # if validate_json_structure(data):
    #     df = pd.DataFrame(data)
    #     return df
    # else:
    #     return pd.DataFrame()

    df = pd.DataFrame(data)
    return df
    # if validate_json_structure(data):
    #     df = pd.DataFrame(data)
    #     return df
    # else:
    #     return pd.DataFrame()


def calculate_token_length(text):
    return len(text.split(" "))


def calculate_f1_recall_precision(reference, predicted):
    ref_tokens = set(reference.split())
    pred_tokens = set(predicted.split())

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


def save_results_to_file(results, filename="./data/results.json"):
    results.to_json(filename, orient="records")


def load_results_from_file(filename="./data/results.json"):
    if os.path.exists(filename):
        return pd.read_json(filename)
    return pd.DataFrame()


def save_results_to_file(results, filename="./data/results.json"):
    results.to_json(filename, orient="records")


def load_results_from_file(filename="./data/results.json"):
    if os.path.exists(filename):
        return pd.read_json(filename)
    return pd.DataFrame()


uploaded_file = st.file_uploader("Upload JSON File", type="json")

df = load_results_from_file()

if uploaded_file:
    df = process_uploaded_file(uploaded_file)
    if not df.empty:
        default_user_prompt = st.text_area(
            "Specify the prompt you want to use (if not provided in the JSON)",
            value="Summarize this document to a max of 2000 words",
        )

        selected_model = st.radio("Select model to benchmark 📈", models)

        insights = df["source"].value_counts().to_frame().reset_index()
        insights.columns = ["Source", "Count"]
        insights["source_flags"] = insights["Source"].map(
            {"mhra": "🇬🇧", "EFSA Journal": "🇪🇺", "SMC": "🇨🇭"}
        )

        grouped_df = df.groupby("source")

        df["tokens_len_full_text"] = df["full-text"].apply(calculate_token_length)
        insights["tokens_len_full_text"] = (
            grouped_df["tokens_len_full_text"].mean().values
        )

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
                    prompt = row.get("prompt", default_user_prompt)
                    predicted = generate_summary(
                        row["full-text"], selected_model, prompt
                    )

                summary_predicted.append(predicted)

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
            df["tokens_len_summary_predicted"] = df["summary_predicted"].apply(
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
            df["selected_model"] = selected_model

            end_time = time()
            total_time = end_time - start_time

            # Rename old results file with timestamp
            if os.path.exists("./data/results.json"):
                timestamp = times.strftime("%Y%m%d-%H%M%S")
                os.rename("./data/results.json", f"./data/_results_{timestamp}.json")

            save_results_to_file(df)
            show = True

    else:
        st.error("The JSON file does not match the required structure.")


try:

    if not df.empty and show:

        scores = []
        for source in df["source"].unique():
            source_data = df[df["source"] == source]
            avg_rouge1 = source_data["ROUGE-1"].median()
            avg_rougeL = source_data["ROUGE-L"].median()
            avg_bleu = source_data["BLEU"].median()
            avg_recall = source_data["Recall"].median()
            avg_precision = source_data["Precision"].median()
            avg_f1 = source_data["F1"].median()
            scores.append(
                {
                    "Model": str(source_data["selected_model"].head(1).values[0]),
                    "Source": source,
                    "ROUGE-1": avg_rouge1,
                    "ROUGE-L": avg_rougeL,
                    "BLEU": avg_bleu,
                    "Recall": avg_recall,
                    "Precision": avg_precision,
                    "F1": avg_f1,
                }
            )
        scores_df = pd.DataFrame(scores)

        st.write("LLM predictions and scores per row:")
        st.write(df)
        st.write("Measures:")
        # st.write(f"It took a total of {total_time} seconds to run the benchmark")
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

except KeyError as e:
    print(f"received the following error: {e}")
