from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


class SentenceEmbedder:
    def __init__(self, model_name: str = 'sentence-transformers/LaBSE'):
        """
        Initialize SentenceEmbedder with a specific SentenceTransformer model.

        Args:
            model_name (str): Name of the SentenceTransformer model. Default is 'sentence-transformers/LaBSE'.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(
            self,
            dataframe: Optional[pd.DataFrame] = None,
            text_varname: str = 'text',
            csv_file_path: Optional[Union[str, Path]] = './data/corpus_sented.csv',
            persist: bool = True,
            output_file_path: Optional[Union[str, Path]] = './work_files/embedding.npy'
    ) -> np.ndarray:
        """
        Generate sentence embeddings for the given DataFrame or CSV file.

        Args:
            dataframe (pd.DataFrame, optional): DataFrame containing the sentences for which embeddings will be created. Default is None.
            text_varname (str): Name of variable containing document text. Default is 'text'.
            csv_file_path (Path or string, optional): File path for the CSV file if dataframe is not provided. Default is './work_files/corpus_sented.csv'.
            persist (bool): Flag determining whether to persist the embeddings to a file. Default is True.
            output_file_path (Path or string, optional): File path for persisting the embeddings. Default is None.

        Returns:
            np.ndarray: Array containing the created sentence embeddings.

        Raises:
            ValueError: If persist is True but output_file_path is not provided.
        """
        if dataframe is None:
            dataframe = pd.read_csv(csv_file_path)

        batch_quantity = 16
        data_count = dataframe.shape[0]

        embeddings = np.zeros((data_count, self.model.get_sentence_embedding_dimension()))

        for idx in tqdm(range(0, data_count, batch_quantity)):
            end_idx = min(idx + batch_quantity, data_count)
            batch_data = dataframe[text_varname][idx:end_idx].tolist()
            batch_embeddings = self.model.encode(batch_data)
            embeddings[idx:end_idx, :] = batch_embeddings

        output_file_path = Path(output_file_path)

        if persist:
            if output_file_path is None:
                raise ValueError("File path must be provided if persist is set to True.")
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file_path, embeddings)

        return embeddings

    def retrieve_embeddings(
            self,
            dataframe: Optional[pd.DataFrame] = None,
            text_varname: str = 'text',
            csv_file_path: Optional[Union[str, Path]] = Path('./work_files/corpus_sented.csv'),
            load_from_file: bool = True,
            embedding_file_path: Optional[Union[str, Path]] = Path('./work_files/embedding'),
            persist: bool = True
    ) -> np.ndarray:
        """
        Retrieve sentence embeddings.

        Args:
            dataframe (pd.DataFrame, optional): DataFrame containing the sentences. Default is None.
            text_varname (str): Name of variable containing document text. Default is 'text'.
            csv_file_path (Path or string, optional): File path for the CSV file if dataframe is not provided. Default is './work_files/corpus_sented.csv'.
            load_from_file (bool): Flag indicating whether to load embeddings from a file. Default is True.
            embedding_file_path (Path or string, optional): File path from where to load the embeddings. Default is None.
            persist (bool): Whether to persist the embeddings to a file. Default is True.

        Returns:
            np.ndarray: Array containing sentence embeddings.

        Raises:
            ValueError: If load_from_file is True but embedding_file_path is not provided.
        """
        embedding_file_path = Path(embedding_file_path)

        if load_from_file:
            if embedding_file_path is None or not embedding_file_path.exists():
                raise FileNotFoundError("Embedding file path must exist if load_from_file is set to True.")
            return np.load(embedding_file_path)
        else:
            return self.generate_embeddings(dataframe, csv_file_path=csv_file_path, persist=persist,
                                            output_file_path=embedding_file_path, text_varname=text_varname)


if __name__ == '__main__':
    embedder = SentenceEmbedder()
    embeddings = embedder.retrieve_embeddings(csv_file_path=Path('../work_files/corpus_sented.csv'),
                                              embedding_file_path=Path('../work_files/embedding'), load_from_file=False)
