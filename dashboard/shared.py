import pickle
from pathlib import Path

import pandas as pd
import polars as pl
import ufal.morphodita
from sentence_transformers import SentenceTransformer

app_dir = Path(__file__).parent

questions_csv_path = "questions_cleaned_filtered.csv"

questions_df = pl.read_csv(questions_csv_path, separator=",")

tagger = ufal.morphodita.Tagger.load(
    "czech-morfflex2.0-pdtc1.0-220710.tagger"
)

with open("rs_filtered.pickle", "rb") as handle:
    rc_dict = pickle.load(handle)


stopwords = []

with open("stopwords-cs.txt", "r") as f:
    for stopword in f:
        stopwords.append(stopword.replace("\n", ""))

PROMPT = """
Mám téma (topic), které se vztahuje k následujícím kvízovým otázkám:
[DOCUMENTS]
Téma lze popsat následujícími klíčovými slovy: [KEYWORDS]
Na základě informací výše, extrahuj krátký popisek tématu, použij češtinu a nepřidávej žádné další informace, délka popisku nechť je mezi 3 a 7 slovy. Popisek uveď v následujícím formátu:
topic: <topic label>
"""

embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
