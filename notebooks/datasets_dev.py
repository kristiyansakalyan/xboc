"""A module that loads all the datasets"""

import re
from typing import Optional, Union

import nltk
import numpy as np
import pandas as pd
from datasets import Split, load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups

try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/wordnet")
except LookupError:
    # If any package is not found, download it
    print("Downloading NLTK packages...")
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")
    print("Download complete.")


def load_20_news_groups():
    data = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes"), data_home="../data"
    )
    docs = data["data"]  # type: ignore
    targets = data["target"]  # type: ignore
    target_names = data["target_names"]  # type: ignore
    classes = [data["target_names"][i] for i in data["target"]]  # type: ignore

    return docs, targets, target_names, classes


def load_bbc_news(split: Optional[Union[str, Split]] = "train"):
    """Loads the BBC News dataset

    Parameters
    ----------
    split : Optional[Union[str, Split]], optional
       "train" or "test" split, by default "train"

    Returns
    -------
    Tuple[List,List,List,Set] - X, y, y_label_text, classes
    """
    ds = load_dataset("SetFit/bbc-news", cache_dir="../data", split=split)
    x = ds["text"]  # type: ignore
    y = ds["label"]  # type: ignore
    y_label_text = ds["label_text"]  # type: ignore
    classes = set(y_label_text)
    return x, y, y_label_text, classes


def load_trump_tweets() -> pd.DataFrame:
    """Returns the Trump's tweets dataset in a Pandas DataFrame

    Returns
    -------
    pd.DataFrame
       The train dataset
    """
    # We only have a train dataset here
    ds = load_dataset("fschlatt/trump-tweets", cache_dir="../data")
    return ds["train"].to_pandas()  # type: ignore


def preprocess_text(text: str, tokenized: bool = False):
    # Lowercasing the text
    text = text.lower()

    # Removing non-alphanumeric characters and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    """
    When training a Word2Vec model, using a lemmatizer like WordNetLemmatizer is generally
    preferred over a stemming algorithm like PorterStemmer because:
        1. Preserves word meanings
        2. Better performance in semantic tasks
    """

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    if tokenized:
        return tokens

    # Joining the tokens back into text
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


def extract_relevant_word_vectors(docs_train, docs_test, idx2word_all, wv_all):

    vocab_corpus_set = set()
    for document in [*docs_train, *docs_test]:
        for word in document:
            if word not in vocab_corpus_set:
                vocab_corpus_set.add(word)
            # else:
            #     print(f'Word was found!\t"{word}"')

    vocab_wv = dict()
    for i, word in enumerate(idx2word_all):
        vocab_wv[word] = i

    wv_filtered = []
    idx2word_filtered = []

    for word in vocab_corpus_set:
        if word in vocab_wv:
            idx2word_filtered.append(word)
            wv_filtered.append(wv_all[vocab_wv[word]])
        # else:
        #     print(f'Word was not found!\t"{word}"')

    # Convert to numpy array
    wv_filtered = np.array(wv_filtered)

    return wv_filtered, idx2word_filtered
