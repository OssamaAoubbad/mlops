import pickle
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from madewithml.config import STOPWORDS


def load_data(dataset_loc: str, num_samples: int = None) -> pd.DataFrame:
    """Load data from source into a pandas DataFrame.

    Args:
        dataset_loc (str): Location of the dataset.
        num_samples (int, optional): The number of samples to load. Defaults to None.

    Returns:
        pd.DataFrame: Our dataset represented by a DataFrame.
    """
    df = pd.read_csv(dataset_loc)
    df = df.sample(frac=1, random_state=1234).reset_index(drop=True)
    if num_samples:
        df = df.head(num_samples).reset_index(drop=True)
    return df


def stratify_split(
    df: pd.DataFrame,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train and test splits with equal
    amounts of data points from each class in the column we
    want to stratify on.

    Args:
        ds (Dataset): Input dataset to split.
        stratify (str): Name of column to split on.
        test_size (float): Proportion of dataset to split for test set.
        shuffle (bool, optional): whether to shuffle the dataset. Defaults to True.
        seed (int, optional): seed for shuffling. Defaults to 1234.

    Returns:
        Tuple[Dataset, Dataset]: the stratified train and test datasets.
    """

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        shuffle=shuffle,
        random_state=seed,
        stratify=df[stratify],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def clean_text(text: str, stopwords: List = STOPWORDS) -> str:
    """Clean raw text string.

    Args:
        text (str): Raw text to clean.
        stopwords (List, optional): list of words to filter out. Defaults to STOPWORDS.

    Returns:
        str: cleaned text.
    """
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # remove links

    return text


def preprocess(df: pd.DataFrame, class_to_index: Dict, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    """Preprocess the data in our dataframe.

    Args:
        df (pd.DataFrame): Raw dataframe to preprocess.
        class_to_index (Dict): Mapping of class names to indices.

    Returns:
        pd.DataFrame: preprocessed data (features, targets).
    """
    outputs = df.copy()
    outputs["title"] = outputs.get("title", "").fillna("")
    outputs["description"] = outputs.get("description", "").fillna("")
    outputs["text"] = (outputs["title"] + " " + outputs["description"]).apply(clean_text)
    features = vectorizer.transform(outputs["text"]).toarray().astype(np.float32)
    outputs = pd.DataFrame({"features": list(features)})
    if "tag" in df.columns:
        outputs["targets"] = df["tag"].map(class_to_index).fillna(-1).astype("int64")
    return outputs


class TextDataset(TorchDataset):
    """PyTorch dataset backed by a dataframe."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]
        item = {"features": row["features"]}
        if "targets" in self.df.columns:
            item["targets"] = row["targets"]
        return item


class CustomPreprocessor:
    """Custom preprocessor class."""

    def __init__(self, class_to_index=None, vectorizer: TfidfVectorizer = None):
        self.class_to_index = class_to_index or {}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.vectorizer = vectorizer or TfidfVectorizer(max_features=5000)

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df["title"] = df.get("title", "").fillna("")
        df["description"] = df.get("description", "").fillna("")
        df["text"] = (df["title"] + " " + df["description"]).apply(clean_text)
        self.vectorizer.fit(df["text"])
        tags = sorted(df["tag"].dropna().unique().tolist())
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, class_to_index=self.class_to_index, vectorizer=self.vectorizer)

    def save(self, path: str) -> None:
        payload = {
            "class_to_index": self.class_to_index,
            "vectorizer": self.vectorizer,
        }
        with open(path, "wb") as fp:
            pickle.dump(payload, fp)

    @classmethod
    def load(cls, path: str) -> "CustomPreprocessor":
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
        return cls(class_to_index=payload["class_to_index"], vectorizer=payload["vectorizer"])
