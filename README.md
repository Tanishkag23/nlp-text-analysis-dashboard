# NLP Text Analysis Dashboard

An interactive dashboard that runs five classic NLP processes over any English
text and reports both the annotated output and the statistics behind it.

Built with **Python · spaCy · Streamlit · pandas · Plotly**.

---

## The five processes

| # | Process | What it does | spaCy component used |
|---|---------|--------------|----------------------|
| 1 | **Tokenization** | Splits raw text into words, numbers and punctuation | `tokenizer` |
| 2 | **Stopword Removal** | Filters out high-frequency function words and punctuation | `Token.is_stop` / `STOP_WORDS` |
| 3 | **POS Tagging** | Labels each token with coarse (UPOS) and fine (Penn Treebank) tags | `tagger`, `attribute_ruler` |
| 4 | **Lemmatization** | Reduces each token to its dictionary base form | `lemmatizer` |
| 5 | **Named Entity Recognition** | Detects people, organisations, places, dates, money, etc. | `ner` |

Each process produces three views:

* **Result** — a token strip plus a full row-by-row table (downloadable as CSV)
* **Statistics** — headline metrics computed separately for that process
* **Visualisation** — a Plotly chart, and spaCy's own `displacy` renderers for
  the entity highlighting and the dependency parse

---

## Quick start (local)

### Option A — one command

**Windows:** double-click `run_windows.bat`
**macOS / Linux:** `./run_mac_linux.sh`

Both scripts create a virtual environment, install everything and launch the app.

### Option B — manual

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies (this also installs the English model)
pip install -r requirements.txt

# 3. if the model did not install, fetch it explicitly
python -m spacy download en_core_web_sm

# 4. run
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

> Requires Python 3.9 – 3.12. First launch downloads roughly 50 MB for the
> spaCy model; after that the app works offline.

---

## How to use it

1. Type or paste text into the input box, or pick one of the four sample
   passages from the sidebar and press **Load sample**.
2. Click any of the five process buttons.
3. Switch between the **Result**, **Statistics** and **Visualisation** tabs.
4. Use **Download result as CSV** if you need the output for a report.

Sidebar toggles let you hide the token strip and change how many table rows
are displayed.

---

## Project structure

```
nlp_dashboard/
├── app.py                 # Streamlit UI: layout, buttons, tabs, charts
├── nlp_engine.py          # All spaCy processing; returns table + stats + chart
├── sample_texts.py        # Demo passages for the sidebar
├── requirements.txt       # Pinned dependencies including en_core_web_sm
├── assets/
│   └── styles.css         # Custom stylesheet (palette, typography, components)
├── .streamlit/
│   └── config.toml        # Streamlit theme matching the stylesheet
├── run_windows.bat        # One-click setup + launch (Windows)
├── run_mac_linux.sh       # One-click setup + launch (macOS / Linux)
└── README.md
```

The separation matters: `nlp_engine.py` contains no Streamlit code, so the
linguistic logic can be tested or reused independently of the interface, and
`app.py` contains no analysis logic.

---

## Statistics reported

| Process | Metrics |
|---------|---------|
| Tokenization | total / word / punctuation / numeric tokens, unique tokens, sentences, characters, type-token ratio, average token length, longest token |
| Stopword Removal | tokens before and after, stopwords and punctuation removed, reduction %, unique stopwords found, size of spaCy's stopword list, cleaned text |
| POS Tagging | tagged tokens, distinct coarse and fine tags, most frequent tag, noun/verb/adjective/adverb counts, lexical density |
| Lemmatization | tokens analysed, lemmas differing from surface form, normalisation rate, unique surface forms vs unique lemmas, vocabulary compression, lemmatised text |
| Named Entity Recognition | entities found, distinct labels, unique entity strings, most frequent label, tokens inside entities, entity coverage, multi-word entities, longest entity |

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to <https://share.streamlit.io>, sign in with GitHub and choose
   **New app**.
3. Select the repository, set the main file to `app.py`, and deploy.

`requirements.txt` already installs the English model from its official wheel,
so no extra build step is needed.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `OSError: [E050] Can't find model 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm` inside the activated virtual environment |
| `streamlit: command not found` | The virtual environment is not active, or use `python -m streamlit run app.py` |
| Fonts look plain | The stylesheet loads Google Fonts; without internet it falls back to system fonts, which is harmless |
| NER returns nothing | The text contains no recognisable entities — try the news sample |
