"""
This module defines a SentenceEmbedder class for generating and retrieving sentence embeddings using the
SentenceTransformer library. The embeddings can be either calculated on the fly or loaded from a persisted numpy file.

The main purpose of this class is to make it easier to manage the process of generating sentence embeddings
for large datasets, including efficient batch processing and optional persistence of the calculated embeddings.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from definitions import project_folders


# noinspection PyTypeChecker
class SentenceEmbedder:
    """
       A class to generate and retrieve sentence embeddings.

       Attributes:
           model (SentenceTransformer): The transformer model used for generating sentence embeddings.

       Methods:
           generate_embeddings: Generate sentence embeddings for a given DataFrame or CSV file.
           retrieve_embeddings: Retrieve sentence embeddings either by generating them or loading from a file.
       """

    def __init__(self, model_name: str = 'sentence-transformers/LaBSE'):
        """
        Initialize SentenceEmbedder with a specific SentenceTransformer model.

        Args:
            model_name: Name of the SentenceTransformer model. Default is 'sentence-transformers/LaBSE'.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(
            self,
            dataframe: Optional[pd.DataFrame] = None,
            text_varname: str = 'text',
            csv_file_path: Optional[Union[str, Path]] = project_folders['work'] / 'corpus_sented.csv',
            persist: bool = True,
            output_file_path: Optional[Union[str, Path]] = project_folders['work'] / 'embedding.npy'
    ) -> np.ndarray:
        """
        Generate sentence embeddings for the given DataFrame or CSV file.

        Args:
            dataframe: DataFrame containing the sentences for which embeddings will be created. Default is None.
            text_varname: Name of variable containing document text. Default is 'text'.
            csv_file_path: File path for the CSV file if dataframe is not provided.
            Default is './work_files/corpus_sented.csv'.
            persist: Flag determining whether to persist the embeddings to a file. Default is True.
            output_file_path: File path for persisting the embeddings. Default is './work_files/embedding.npy'.

        Returns:
            Array containing the created sentence embeddings.

        Raises:
            ValueError: If persist is True but output_file_path is not provided.
        """
        if dataframe is None:
            dataframe = pd.read_csv(csv_file_path)

        batch_quantity = 16
        data_count = dataframe.shape[0]
        sentence_embeddings = np.zeros((data_count, self.model.get_sentence_embedding_dimension()))

        for idx in tqdm(range(0, data_count, batch_quantity)):
            end_idx = min(idx + batch_quantity, data_count)
            batch_data = dataframe[text_varname][idx:end_idx].tolist()
            batch_embeddings = self.model.encode(batch_data)
            sentence_embeddings[idx:end_idx, :] = batch_embeddings

        if persist:
            try:
                output_file_path = Path(output_file_path)
            except(TypeError):
                raise ValueError("File path must be provided if persist is set to True.")
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file_path, sentence_embeddings)

        return sentence_embeddings

    def retrieve_embeddings(
            self,
            dataframe: Optional[pd.DataFrame] = None,
            text_varname: str = 'text',
            csv_file_path: Union[str, Path] = project_folders['work'] / 'corpus_sented.csv',
            load_from_file: bool = False,
            embedding_file_path: Union[str, Path] = project_folders['work'] / 'embedding',
            persist: bool = True
    ) -> np.ndarray:
        """
        Retrieve sentence embeddings.

        Args:
            dataframe: DataFrame containing the sentences. Default is None.
            text_varname: Name of variable containing document text. Default is 'text'.
            csv_file_path: File path for the CSV file if dataframe is not provided.
            Default is './work_files/corpus_sented.csv'.
            load_from_file: Flag indicating whether to load embeddings from a file. Default is False.
            embedding_file_path: File path from where to load the embeddings. Default is './work_files/embedding'.
            persist: Whether to persist the embeddings to a file. Default is True.

        Returns:
            Array containing sentence embeddings.

        Raises:
            ValueError: If load_from_file is True but embedding_file_path is not provided.
        """
        if load_from_file:
            try:
                embedding_file_path = Path(embedding_file_path)
                if not embedding_file_path.exists():
                    raise FileNotFoundError('Embedding file path must exist if load_from_file is set to True.')
            except(TypeError):
                raise TypeError('Embedding file path must be provided if load_from_file is set to true')
            return np.load(embedding_file_path)
        else:
            return self.generate_embeddings(dataframe, csv_file_path=csv_file_path, persist=persist,
                                            output_file_path=embedding_file_path, text_varname=text_varname)


if __name__ == '__main__':
    embedder = SentenceEmbedder()
    embeddings = embedder.retrieve_embeddings(csv_file_path=project_folders['work'] / 'corpus_brand_sented.csv',
                                              embedding_file_path=project_folders['work'] / 'embedding_brand.npy',
                                              load_from_file=False)
