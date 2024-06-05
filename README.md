## Streamlit app for the Benchmark

### How to run it

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nltk.download('punkt')
source .env
streamlit run app.py
```



## ToDo


- [x] remove table of uploads
- [x] average token lenghts
- [x] model selection
- [x] add to the blue score
- [x] add recall, precision and f1
- [ ] paragraph similarity -> sentence transformers / sentencepiece library -> https: //sbert.net/ --> cosine