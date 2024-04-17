from pathlib import Path
from typing import List, Optional

import hu_core_news_lg
import pandas as pd
from tqdm import tqdm

from definitions import project_folders

tqdm.pandas()
nlp = hu_core_news_lg.load()


def split_into_sentences(text: str) -> List[str]:
    """
    Splits the input text into sentences using HuSpacy's hu_core_news_lg model.

    Args:
        text (str): The input text.

    Returns:
        List[str]: The list of sentences.
    """
    return [sentence.text.strip() for sentence in nlp(text).sents]


def sentencize_dataframe(
        dataframe: Optional[pd.DataFrame] = None,
        persist: bool = True,
        text_column: str = 'text',
        save_path: str = project_folders['work'] / 'corpus_sented.csv',
        document_columns: List[str] = None
) -> pd.DataFrame:
    """
    Splits the text in the DataFrame into sentences and return a new DataFrame in exploded format.

    Args:
        dataframe (Optional[pd.DataFrame]): The input DataFrame containing 'cat_id', 'general_cat', and 'text' columns.
            Default is None.
        persist (bool): Whether to save the resulting DataFrame to a file. Default is True.
        text_column (str): The name of the column in the DataFrame containing the text. Default is 'text'.
        save_path (str): Path to save sentenced DataFrame. Default is project_folders['work'] / '/corpus_sented.csv'.
        document_columns (List[str]): List of category column names. Default is None.

    Returns:
        pd.DataFrame: The new DataFrame with 'doc_id', 'id', 'text', and 'sentences' columns.
    """

    if dataframe is None:
        dataframe = pd.read_csv(project_folders['work'] / '/corpus.csv')

    if document_columns is None:
        document_columns = [col for col in list(dataframe.columns) if not col == 'text_column']

    sentences_dataframe = pd.merge(
        dataframe[document_columns],
        dataframe[text_column].progress_apply(split_into_sentences).explode().rename('sentences'),
        left_index=True,
        right_index=True
    ).reset_index(drop=True)

    sentences_dataframe = sentences_dataframe[sentences_dataframe.sentences != '']

    if persist:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        sentences_dataframe.to_csv(save_path, index=False)

    return sentences_dataframe


if __name__ == '__main__':
    sentencize_dataframe()
