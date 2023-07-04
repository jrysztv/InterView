import itertools
from pathlib import Path
from typing import List, Union, Mapping, Optional, Dict

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from umap import UMAP

from utils.helpers import save_xls, get_stopwords


class BERTopicModelling:
    def __init__(self, sentence_transformer_name: str, result_dir: str):
        """
        Initialize BERTopicModelling class.

        Args:
            sentence_transformer_name (str): Name of the sentence transformer model.
            result_dir (str): Directory where the results will be saved.
        """
        self.model = SentenceTransformer(sentence_transformer_name)
        self.result_path = Path('../') / Path(result_dir)
        self.result_path.mkdir(parents=True, exist_ok=True)

    def estimate_topic_model(self, text_instances: Union[pd.Series, List], embeddings: np.array,
                             umap_params: Mapping[str, float | int | str | List],
                             hdbscan_params: Mapping[str, float | int | str | List],
                             vectorizer_params: Dict[str, List[str]]) -> tuple[
        list[int], ndarray | ndarray | None, BERTopic]:
        """
        Estimates a BERTopic model.

        Args:
            text_instances (pd.Series): Series of text instances like sentences or paragraphs.
            embeddings (np.array): Sentence embeddings.
            umap_params (dict): Parameters for the UMAP model.
            hdbscan_params (dict): Parameters for the HDBSCAN model.
            vectorizer_params (dict): Parameters for the CountVectorizer.

        Returns:
            BERTopic: Estimated BERTopic model.
        """
        umap_model = UMAP(**umap_params)
        hdbscan_model = HDBSCAN(**hdbscan_params)
        vectorizer_model = CountVectorizer(**vectorizer_params)

        topic_model = BERTopic(embedding_model=self.model,
                               umap_model=umap_model,
                               hdbscan_model=hdbscan_model,
                               vectorizer_model=vectorizer_model)
        topics, probs = topic_model.fit_transform(text_instances, embeddings)

        return topics, probs, topic_model

    def iterate_and_save_over_categories(self, df: pd.DataFrame, embeddings: np.array, stopwords: list,
                                         param_grid: list, category_variable: Optional[str] = 'general_cat'):
        for category in tqdm(df[category_variable].unique()):
            if df[df[category_variable] == category].shape[0] >= 100 and not (
                    self.result_path / f"{category}.xlsx").exists():
                topic_tables = self.apply_param_grid_to_topic_model(df, embeddings, stopwords, param_grid)
                save_xls(topic_tables, self.result_path / "topic_models_by_category" / f"{category}.xlsx")

    def apply_param_grid_to_topic_model(self, corpus_dataframe: pd.DataFrame, embeddings: np.array,
                                        stopwords: list,
                                        param_grid: list,
                                        text_varname: str = 'sentences'):
        """
        Applies a parameter grid to the bertopic modelling framework.

        Args:
            text_varname ():
            corpus_dataframe ():
            embeddings (np.array): Sentence embeddings.
            stopwords (list): List of stopwords. (Should be Hungarian).
            param_grid (list): Parameter grid to optimize over.
        """
        text_entities = corpus_dataframe[text_varname]
        vectorizer_params = {"stop_words": stopwords}
        topic_tables = []
        for params in tqdm(param_grid):
            umap_params, hdbscan_params = params
            _, __, topic_model = self.estimate_topic_model(text_entities, embeddings,
                                                           umap_params, hdbscan_params, vectorizer_params)
            topic_tables.append(topic_model.get_topic_info())
        return topic_tables

    def estimate_and_save_one_result(self, sentencized_dataframe: pd.DataFrame, embeddings: np.array,
                                     umap_params: Mapping[str, List[float | int | str] | float | int | str],
                                     hdbscan_params: Mapping[str, List[float | int | str] | float | int | str],
                                     vectorizer_params: Mapping[str, List[str]],
                                     filename: str = 'topic_model',
                                     text_varname: str = 'sentences'):
        """
        Estimates one BERTopic model and saves it to an Excel file.

        Args:
            filename ():
            vectorizer_params ():
            hdbscan_params ():
            umap_params ():
            sentencized_dataframe (pd.DataFrame): Dataframe of sentences.
            embeddings (np.array): Sentence embeddings.
        """
        _, __, full_model = self.estimate_topic_model(sentencized_dataframe[text_varname], embeddings, umap_params,
                                                      hdbscan_params,
                                                      vectorizer_params)
        full_model.get_topic_info().to_excel(self.result_path / f'{filename}.xlsx')

    def save_param_grid_results(self, topic_tables: List[pd.DataFrame], filename: str = 'full_param_grid'):
        save_xls(topic_tables, self.result_path / f'{filename}.xlsx')


if __name__ == "__main__":
    # Instantiate the BERTopicModelling class
    print('loading_model')
    bertopic_modeller = BERTopicModelling('sentence-transformers/LaBSE', './results')

    # Get Hungarian stopwords
    print('getting stopwords')
    hu_stopwords = get_stopwords()

    # Load your data and embeddings here
    print('importing')
    sentencized_corpus = pd.read_csv('../work_files/corpus_sented.csv')
    sentence_embeddings = np.load('../work_files/embedding_sentences.npy')

    # Estimate and save full model with fixed parameters
    print('param_grid')
    umap_params = {'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.00, 'metric': 'cosine'}
    hdbscan_params = {'min_cluster_size': 15, 'metric': 'euclidean', 'cluster_selection_method': 'eom',
                      'prediction_data': True}
    vectorizer_params = {"stop_words": hu_stopwords}

    bertopic_modeller.estimate_and_save_one_result(sentencized_corpus, sentence_embeddings, umap_params, hdbscan_params,
                                                   vectorizer_params)

    # Set up parameter grid
    umap_params_list = [{'n_neighbors': n, 'n_components': c, 'min_dist': d, 'metric': 'cosine'}
                        for n in [5, 20]
                        for c in [2, 3, 5]
                        for d in [.1]]
    hdbscan_params_list = [
        {'min_cluster_size': m, 'metric': 'euclidean', 'cluster_selection_method': 'eom', 'prediction_data': True} for m
        in [5, 15, 30, 50]]
    param_grid = list(itertools.product(umap_params_list, hdbscan_params_list))

    print(param_grid)

    # Optimize topic model
    topic_tables = bertopic_modeller.apply_param_grid_to_topic_model(sentencized_corpus,
                                                                     sentence_embeddings,
                                                                     hu_stopwords, param_grid)
    bertopic_modeller.save_param_grid_results(topic_tables)
