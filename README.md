## Streamlit app for the Benchmark

### How to run it

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nltk.download('punkt')
streamlit run app.py
```

### Prompt

Create a streamlit application with the Header "Benchmark your LLM on creating a Summary". There should be an infotext, explaining what the tool does. Thereafter there should be a fileupload that accepts only JSON documents following the structre [
    {
        "full-text": "Blbablablabla",
        "summary": "bla",
        "source": "MHRA / EFSA / SMC",
        "URL": "someabazingurl.com"
    }
]

Raise an error if it is not this structure.

If the file is uplaoded successfully, show the table with the the content and provide some insside ghow many entries were there per source. Use the flags icons (UK for MHRA, CH for SMC and EU flat for EFSA). Below the table there is a button called "run benchmark". Doing so would call an openai endpoint and compute a new column to the dataset called summary_predicted. calculate how long the benchmark will take. Thereafter, provide per source a report on the time it took to run the benchmark, the rogue score of summary_predicted and summary, as wel l as the blue score.