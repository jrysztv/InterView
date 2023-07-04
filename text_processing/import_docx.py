import os
from typing import List

import pandas as pd
from docx import Document


class DocumentToDataFrame:
    def __init__(self, data_folder: str = './data/') -> None:
        """
        Initialize the DocumentToDataFrame class.

        Args:
            data_folder (str): Path to the folder containing the data files. Default is './data/'.
        """
        self.data_folder = data_folder if data_folder.endswith(('/', '\\')) else data_folder + '/'

    def convert_docs_to_strings(self) -> pd.DataFrame:
        """
        Convert docx files to a string representation.

        Returns:
            pd.DataFrame: A DataFrame containing the converted texts with corresponding document IDs.
        """
        document_filepaths: List[str] = [os.path.join(self.data_folder, filename) for filename in
                                         os.listdir(self.data_folder) if
                                         filename.endswith('.docx')]
        document_texts: List[str] = ['\n'.join([par.text for par in Document(file).paragraphs]).strip() for file in
                                     document_filepaths]
        document_corpus: pd.DataFrame = pd.DataFrame(
            pd.Series(document_texts, name='texts')
            .str.split('\n').explode()
        ).reset_index(names='doc_id')
        return document_corpus

    def filter_relevant_text(self, document_corpus: pd.DataFrame) -> pd.DataFrame:
        """
        Filter relevant strings based on a specific logic and convert them into a new DataFrame.

        Args:
            document_corpus (pd.DataFrame): Input corpus DataFrame.

        Returns:
            pd.DataFrame: Filtered corpus DataFrame with extracted IDs and relevant texts.
        """
        document_corpus = document_corpus[document_corpus.texts.str.count(':') > 0]
        document_corpus = pd.concat(
            [document_corpus['doc_id'],
             document_corpus['texts'].str.split(':', n=1)
             .apply(pd.Series)
             .rename(columns={0: 'id', 1: 'text'})],
            axis=1
        )
        document_corpus['text'] = document_corpus['text'].str.extract(r' ?-? ?(.*)')
        document_corpus = document_corpus[document_corpus['text'] != '']
        return document_corpus

    def clean_identifiers(self, document_corpus: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the IDs in the document corpus DataFrame.

        Args:
            document_corpus (pd.DataFrame): Input corpus DataFrame.

        Returns:
            pd.DataFrame: Corpus DataFrame with cleaned IDs.
        """
        invalid_ids = ['[Valaki felsóhajt', '[Kórusban ]', '[női hang]', '[férfi hang]', '[Férfihang]',
                       '[többen összesúgnak',
                       '[Valaki suttogva', '[Viccből]']
        document_corpus = document_corpus[document_corpus.id.str.count(' ') < 2]
        document_corpus = document_corpus[~document_corpus.id.isin(invalid_ids)]
        document_corpus['id'] = document_corpus['id'].str.replace('[^a-zA-ZáéíóöőüűÁÉÍÓÖŐÜŰ\s]', '', regex=True)
        document_corpus = document_corpus[document_corpus['id'] != '']
        return document_corpus

    def generate_corpus(self) -> pd.DataFrame:
        """
        Get the final processed corpus.

        Returns:
            pd.DataFrame: Processed corpus DataFrame.
        """
        document_corpus = self.convert_docs_to_strings()
        document_corpus = self.filter_relevant_text(document_corpus)
        document_corpus = self.clean_identifiers(document_corpus)
        return document_corpus


if __name__ == '__main__':
    corpus_generator = DocumentToDataFrame(data_folder='../data')
    corpus = corpus_generator.generate_corpus()
    corpus.to_csv('../work_files/corpus.csv', index=False)
