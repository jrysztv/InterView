from pathlib import Path
from typing import List

import hu_core_news_lg
import numpy as np
import pandas as pd
from pandas import ExcelWriter


def get_stopwords() -> list:
    """
    Gets a list of Hungarian stopwords.

    Returns:
        list: List of Hungarian stopwords.
    """
    nlp = hu_core_news_lg.load()
    hu_sw = list(nlp.Defaults.stop_words)
    hu_sw.extend(['szeretnék', 'szeretném', 'szeret'])
    return hu_sw


def cat_subset(df: pd.DataFrame, embeddings: np.array, cat_name: str = 'general_cat', cat: str = 'NTP-ADY') -> tuple:
    """
    Gets a subset of sentences and embeddings corresponding to a specific category.

    Args:
        df (pd.DataFrame): Dataframe of sentences.
        embeddings (np.array): Sentence embeddings.
        cat (str): The category to subset on. Default is 'NTP-ADY'.

    Returns:
        tuple: A tuple of (sentences, embeddings) for the specific category.
    """
    df = df[df[cat_name] == cat]
    return df['text'], embeddings[df.index.values]


def save_xls(list_dfs: List[pd.DataFrame], xls_path: Path):
    """
    Saves a list of dataframes to an Excel file, each dataframe in a separate sheet.

    Args:
        list_dfs (List[pd.DataFrame]): List of dataframes.
        xls_path (Path): Path to save the Excel file.
    """
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, 'sheet%s' % n)
