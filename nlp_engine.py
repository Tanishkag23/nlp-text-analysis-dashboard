"""
nlp_engine.py
-------------
All linguistic processing for the NLP Text Analysis Dashboard.

Every function takes a spaCy `Doc` and returns a dictionary with three keys:

    "table"  -> pandas.DataFrame  : the row-by-row result of the process
    "stats"  -> dict              : headline numbers shown on the Statistics tab
    "chart"  -> pandas.DataFrame  : a small frame ready to plot (may be None)

Keeping the analysis here (and out of app.py) means the UI layer only has to
worry about presentation, which makes both files easy to read and to mark.
"""

from collections import Counter

import pandas as pd
import spacy
from spacy.cli import download as spacy_download
from spacy.lang.en.stop_words import STOP_WORDS

MODEL_NAME = "en_core_web_sm"


# --------------------------------------------------------------------------- #
#  Model loading
# --------------------------------------------------------------------------- #
def load_model(model_name: str = MODEL_NAME):
    """Load the spaCy pipeline, downloading it once if it is missing."""
    try:
        return spacy.load(model_name)
    except OSError:
        spacy_download(model_name)
        return spacy.load(model_name)


def analyse(nlp, text: str):
    """Run the text through the pipeline a single time and reuse the Doc."""
    return nlp(text)


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #
def _pct(part: int, whole: int) -> float:
    return round((part / whole) * 100, 2) if whole else 0.0


def _counter_frame(counter: Counter, key_name: str, top=None):
    items = counter.most_common(top) if top else counter.most_common()
    return pd.DataFrame(items, columns=[key_name, "Count"])


# --------------------------------------------------------------------------- #
#  1. Tokenization
# --------------------------------------------------------------------------- #
def tokenize(doc):
    rows = []
    for i, token in enumerate(doc, start=1):
        rows.append(
            {
                "#": i,
                "Token": token.text,
                "Type": (
                    "Punctuation"
                    if token.is_punct
                    else "Number"
                    if token.like_num
                    else "Word"
                    if token.is_alpha
                    else "Other"
                ),
                "Characters": len(token.text),
                "Shape": token.shape_,
                "Is stopword": token.is_stop,
            }
        )
    table = pd.DataFrame(rows)

    words = [t for t in doc if t.is_alpha]
    unique = {t.text.lower() for t in doc if not t.is_punct and not t.is_space}
    lengths = [len(t.text) for t in words]

    stats = {
        "Total tokens": len(doc),
        "Word tokens": len(words),
        "Punctuation tokens": sum(1 for t in doc if t.is_punct),
        "Numeric tokens": sum(1 for t in doc if t.like_num),
        "Unique tokens": len(unique),
        "Sentences": len(list(doc.sents)),
        "Characters (with spaces)": len(doc.text),
        "Type-token ratio": round(len(unique) / max(len(words), 1), 3),
        "Average token length": round(sum(lengths) / max(len(lengths), 1), 2),
        "Longest token": max(words, key=lambda t: len(t.text)).text if words else "—",
    }

    length_counts = Counter(lengths)
    chart = pd.DataFrame(
        sorted(length_counts.items()), columns=["Token length", "Count"]
    )
    return {"table": table, "stats": stats, "chart": chart}


# --------------------------------------------------------------------------- #
#  2. Stopword removal
# --------------------------------------------------------------------------- #
def remove_stopwords(doc):
    rows, kept = [], []
    for i, token in enumerate(doc, start=1):
        if token.is_space:
            continue
        removed = token.is_stop or token.is_punct
        reason = (
            "Stopword" if token.is_stop else "Punctuation" if token.is_punct else "—"
        )
        if not removed:
            kept.append(token.text)
        rows.append(
            {
                "#": i,
                "Token": token.text,
                "Status": "Removed" if removed else "Kept",
                "Reason": reason,
            }
        )
    table = pd.DataFrame(rows)

    considered = len(table)
    removed_tokens = [t for t in doc if t.is_stop]
    stats = {
        "Tokens before": considered,
        "Tokens after": len(kept),
        "Stopwords removed": len(removed_tokens),
        "Punctuation removed": sum(1 for t in doc if t.is_punct),
        "Reduction": f"{_pct(considered - len(kept), considered)} %",
        "Unique stopwords found": len({t.text.lower() for t in removed_tokens}),
        "Stopwords in spaCy list": len(STOP_WORDS),
        "Cleaned text": " ".join(kept),
    }

    chart = _counter_frame(
        Counter(t.text.lower() for t in removed_tokens), "Stopword", top=12
    )
    return {"table": table, "stats": stats, "chart": chart}


# --------------------------------------------------------------------------- #
#  3. Part-of-speech tagging
# --------------------------------------------------------------------------- #
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def pos_tag(doc):
    rows = []
    for i, token in enumerate(doc, start=1):
        if token.is_space:
            continue
        rows.append(
            {
                "#": i,
                "Token": token.text,
                "Coarse tag (POS)": token.pos_,
                "Fine tag (TAG)": token.tag_,
                "Description": spacy.explain(token.tag_) or "—",
                "Dependency": token.dep_,
                "Head": token.head.text,
            }
        )
    table = pd.DataFrame(rows)

    counts = Counter(t.pos_ for t in doc if not t.is_space)
    content = sum(v for k, v in counts.items() if k in CONTENT_POS)
    total = sum(counts.values())

    stats = {
        "Tagged tokens": total,
        "Distinct coarse tags": len(counts),
        "Distinct fine tags": len({t.tag_ for t in doc if not t.is_space}),
        "Most frequent tag": f"{counts.most_common(1)[0][0]} ({counts.most_common(1)[0][1]})"
        if counts
        else "—",
        "Nouns": counts.get("NOUN", 0) + counts.get("PROPN", 0),
        "Verbs": counts.get("VERB", 0) + counts.get("AUX", 0),
        "Adjectives": counts.get("ADJ", 0),
        "Adverbs": counts.get("ADV", 0),
        "Lexical density": f"{_pct(content, total)} %",
    }

    chart = _counter_frame(counts, "POS tag")
    chart["Meaning"] = chart["POS tag"].apply(lambda p: spacy.explain(p) or "—")
    return {"table": table, "stats": stats, "chart": chart}


# --------------------------------------------------------------------------- #
#  4. Lemmatization
# --------------------------------------------------------------------------- #
def lemmatize(doc):
    rows = []
    for i, token in enumerate(doc, start=1):
        if token.is_space:
            continue
        changed = token.text.lower() != token.lemma_.lower()
        rows.append(
            {
                "#": i,
                "Token": token.text,
                "Lemma": token.lemma_,
                "POS": token.pos_,
                "Changed": "Yes" if changed else "No",
            }
        )
    table = pd.DataFrame(rows)

    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    changed_rows = [
        t for t in tokens if t.text.lower() != t.lemma_.lower()
    ]
    unique_tokens = {t.text.lower() for t in tokens}
    unique_lemmas = {t.lemma_.lower() for t in tokens}

    stats = {
        "Tokens analysed": len(tokens),
        "Lemmas differing from token": len(changed_rows),
        "Unchanged tokens": len(tokens) - len(changed_rows),
        "Normalisation rate": f"{_pct(len(changed_rows), len(tokens))} %",
        "Unique surface forms": len(unique_tokens),
        "Unique lemmas": len(unique_lemmas),
        "Vocabulary compression": f"{_pct(len(unique_tokens) - len(unique_lemmas), len(unique_tokens))} %",
        "Lemmatised text": " ".join(t.lemma_ for t in doc if not t.is_space),
    }

    chart = _counter_frame(
        Counter(t.lemma_.lower() for t in tokens if not t.is_stop), "Lemma", top=12
    )
    return {"table": table, "stats": stats, "chart": chart}


# --------------------------------------------------------------------------- #
#  5. Named entity recognition
# --------------------------------------------------------------------------- #
def named_entities(doc):
    rows = []
    for i, ent in enumerate(doc.ents, start=1):
        rows.append(
            {
                "#": i,
                "Entity": ent.text,
                "Label": ent.label_,
                "Description": spacy.explain(ent.label_) or "—",
                "Start char": ent.start_char,
                "End char": ent.end_char,
                "Tokens": len(ent),
            }
        )
    table = pd.DataFrame(
        rows,
        columns=["#", "Entity", "Label", "Description", "Start char", "End char", "Tokens"],
    )

    counts = Counter(ent.label_ for ent in doc.ents)
    tokens_in_ents = sum(len(ent) for ent in doc.ents)
    total_tokens = sum(1 for t in doc if not t.is_space)

    stats = {
        "Entities found": len(doc.ents),
        "Distinct labels": len(counts),
        "Unique entity strings": len({ent.text.lower() for ent in doc.ents}),
        "Most frequent label": f"{counts.most_common(1)[0][0]} ({counts.most_common(1)[0][1]})"
        if counts
        else "—",
        "Tokens inside entities": tokens_in_ents,
        "Entity coverage": f"{_pct(tokens_in_ents, total_tokens)} %",
        "Multi-word entities": sum(1 for ent in doc.ents if len(ent) > 1),
        "Longest entity": max(doc.ents, key=len).text if doc.ents else "—",
    }

    chart = _counter_frame(counts, "Entity label")
    if not chart.empty:
        chart["Meaning"] = chart["Entity label"].apply(
            lambda label: spacy.explain(label) or "—"
        )
    return {"table": table, "stats": stats, "chart": chart}


# --------------------------------------------------------------------------- #
#  Registry used by the dashboard
# --------------------------------------------------------------------------- #
PROCESSES = {
    "Tokenization": {
        "fn": tokenize,
        "blurb": "Splits the text into tokens — words, numbers and punctuation — using spaCy's rule-based tokenizer.",
        "empty": "No tokens were produced.",
    },
    "Stopword Removal": {
        "fn": remove_stopwords,
        "blurb": "Filters out high-frequency function words and punctuation that carry little topical meaning.",
        "empty": "Every token survived the filter.",
    },
    "POS Tagging": {
        "fn": pos_tag,
        "blurb": "Labels each token with its part of speech, both the coarse Universal tag and the fine-grained Penn Treebank tag.",
        "empty": "No tokens were tagged.",
    },
    "Lemmatization": {
        "fn": lemmatize,
        "blurb": "Reduces each token to its dictionary base form using spaCy's lookup and rule-based lemmatizer.",
        "empty": "No tokens were lemmatised.",
    },
    "Named Entity Recognition": {
        "fn": named_entities,
        "blurb": "Detects real-world entities such as people, organisations, places, dates and money amounts.",
        "empty": "No named entities were detected in this text. Try a passage that mentions people, places, organisations or dates.",
    },
}
