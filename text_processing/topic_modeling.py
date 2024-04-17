"""
This module provides a class BERTopicModelling which performs topic modelling on textual data using BERT transformers.
"""

import itertools
import pickle
from builtins import list
from pathlib import Path
from typing import List, Union, Mapping, Optional, Dict, Tuple, TypeVar, Any, Type

import numpy as np
import pandas as pd
from bertopic import BERTopic  # TODO: write the code that imports and applies HDBSCAN, countvec and umap
from hdbscan import HDBSCAN
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from umap import UMAP

from definitions import project_folders
from utils.helpers import save_xls, get_stopwords

T = TypeVar('T')


class ParamContainer:
    def __init__(self, sentence_transformer_name: str = 'sentence-transformers/LaBSE',
                 stopwords: Optional[List[str]] = None,
                 dimensionality_reducer_params: Optional[Dict[str, Any]] = None,
                 vectorizer_params: Optional[Dict[str, Any]] = None,
                 clusterer_params: Optional[Dict[str, Any]] = None,
                 sentence_transformer_type: Type[T] = SentenceTransformer,
                 vectorizer_type: Type[T] = CountVectorizer,
                 dimensionality_reducer_type: Type[T] = UMAP,
                 clusterer_type: Type[T] = HDBSCAN):
        # Model Params
        self.sentence_transformer_type = sentence_transformer_type
        self.sentence_transformer_name = sentence_transformer_name
        self.vectorizer_type = vectorizer_type
        self.stopwords = stopwords if stopwords is not None else get_stopwords()
        self.vectorizer_params = vectorizer_params if vectorizer_params is not None else {'stop_words': stopwords}
        self.dimensionality_reducer_type = dimensionality_reducer_type
        self.dimensionality_reducer_params = dimensionality_reducer_params
        self.clusterer_type = clusterer_type
        self.clusterer_params = clusterer_params

        # Defaults
        self._default_sentence_transformer_name = self.sentence_transformer_name
        self.default_vectorizer_params = self.vectorizer_params
        self.default_dimensionality_reducter_params = self.dimensionality_reducer_params
        self.default_clusterer_params = self.clusterer_params


class ModelContainer:
    """
    Model Container. Param container needs to be initialized before.
    """

    def __init__(self, param_container: ParamContainer = ParamContainer()):

        self.param_set = param_container

        # Models
        self.sentence_transformer = None
        self.dimensionality_reducer = None
        self.vectorizer = None
        self.clusterer = None

        # Set Models
        self.set_models()

    def set_models(self):
        self.sentence_transformer = self.create_model(self.param_set.sentence_transformer_type,
                                                      {'model_name_or_path': self.param_set.sentence_transformer_name})
        self.dimensionality_reducer = self.create_model(self.param_set.dimensionality_reducer_type,
                                                        self.param_set.dimensionality_reducer_params,
                                                        self.param_set.default_dimensionality_reducter_params)
        self.vectorizer = self.create_model(self.param_set.vectorizer_type, self.param_set.vectorizer_params,
                                            self.param_set.default_vectorizer_params)
        self.clusterer = self.create_model(self.param_set.clusterer_type, self.param_set.clusterer_params,
                                           self.param_set.default_clusterer_params)

    def create_model(self, ModelClass: Type[T], params: Optional[Dict[str, Any]] = None,
                     default_params: Optional[Dict[str, Any]] = None) -> T:
        """
        Create an instance of a specified class using the provided parameters.

        Args:
            ModelClass ():
            default_params (Optional[Dict[str, Any]):
            params (Optional[Dict[str, Any]):
        """
        params = params or default_params or {}
        try:
            return ModelClass(**params)
        except:
            return ModelClass()


class EmbeddingsContainer:
    def __init__(self, embeddings: Optional[np.array] = None):  # TODO: add read method - after first release
        self.embeddings = embeddings
        self.default_embeddings = embeddings

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, embeddings_array: np.array):
        self._embeddings = embeddings_array

    def reset_embeddings(self):
        """
        Method to reset embeddings to the one saved upon initialization.
        """
        self.embeddings = self.default_embeddings


class TextContainer:
    """
    Contains the Series of strings to be subjected to Topic Modelling
    """

    def __init__(self, corpus_input: Union[pd.DataFrame, pd.Series, List[str]] = None,
                 text_varname: Optional[str] = 'text'):
        self.text_varname = text_varname
        self.text_instances = corpus_input

    @property
    def text_varname(self):
        return self._text_varname

    @text_varname.setter
    def text_varname(self, value: str):
        self._text_varname = value

    @property
    def text_instances(self):
        return self._text_instances

    @text_instances.setter
    def text_instances(self, corpus_input: Union[pd.DataFrame, pd.Series, List[str]]):
        if isinstance(corpus_input, pd.DataFrame):
            if self.text_varname is None:
                raise ValueError("No text variable name was provided for the DataFrame.")
            if self.text_varname not in corpus_input.columns:
                raise ValueError(f'Text variable name "{self.text_varname}" not found in the DataFrame columns.')
            self._text_instances = corpus_input[self.text_varname]
        else:
            self._text_instances = corpus_input


class EmptyUMAP:
    """
    Helper class for the BERTopicModellingOptimized class. Return embeddings unchanged.
    Yields a workaround solution by supplying the already dimensionality-reduced embeddings to the BERTopic pipeline.
    """

    def fit(self, x, y):
        return self

    def transform(self, embeddings):
        return embeddings


class TopicEstimator:
    """
    A class for topic modeling with BERT transformers..
    """

    def __init__(self,
                 text_container: TextContainer,
                 embeddings_container: EmbeddingsContainer,
                 applied_models: ModelContainer,
                 ):
        """
        Initialize a BERTopicModelling object with a specific transformer model and a result directory.

        Args:
        """

        self.text_instances = text_container.text_instances
        self.embeddings_container = embeddings_container
        # ^^ Using both _default and mutable embeddings, referring to object for clarity

        self.sentence_transformer = applied_models.sentence_transformer
        self.dimensionality_reducer = applied_models.dimensionality_reducer
        self.vectorizer = applied_models.vectorizer
        self.clusterer = applied_models.clusterer

        self.probs = None
        self.topics = None
        self.topic_model = None

    def estimate_topic_model(self) -> Tuple[List[int], Optional[np.ndarray], BERTopic]:
        """
        Estimates a BERTopic model using the object attributes set via setter methods.
        """

        topic_model = BERTopic(embedding_model=self.sentence_transformer,
                               umap_model=self.dimensionality_reducer,
                               hdbscan_model=self.clusterer,
                               vectorizer_model=self.vectorizer)
        topics, probs = topic_model.fit_transform(self.text_instances, self.embeddings_container.embeddings)
        self.topics = topics
        self.probs = probs
        self.topic_model = topic_model

        return topics, probs, topic_model


# Todo: create Results class to store and save model results

class IterateTopicEstimator():
    # TODO: This part should use the EmptyUMAP - https://github.com/MaartenGr/BERTopic/issues/491
    # Another useful link: https://github.com/MaartenGr/BERTopic/issues/278
    """
    Applies the TopicEstimator instance and saves the results.
    The class is able to apply a param grid for estimation or perform an optimized iteration of first reducing
    dimensionality and then performing clustering over the distinguished list of parameter sets
    """

    def __init__(self,
                 param_container: ParamContainer,  # Todo: abbreviate class names (without container)
                 embeddings_container: EmbeddingsContainer,
                 text_container: TextContainer,
                 filename: Optional[str] = 'topic_model.xlsx',
                 paramgrid_filename: Optional[str] = 'full_param_grid.xlsx',
                 reduced_embeddings_folder='reduced_embeddings',
                 dimensionality_reducer_params_to_iterate: Optional[list[Dict[str, Any]]] = None,
                 clusterer_params_to_iterate: Optional[list[Dict[str, Any]]] = None,
                 param_grid: Optional[List[Dict[str, Any]]] = None
                 ):

        # Filepaths
        self.result_path = project_folders['result']
        self.single_result_path = project_folders['result'] / filename
        self.paramgrid_result_path = project_folders['result'] / paramgrid_filename
        self.reduced_embeddings_path = project_folders[
                                           'work'] / reduced_embeddings_folder  # TODO: set as property with setters :)
        self.reduced_embeddings_path.mkdir(parents=True, exist_ok=True)  # TODO: put this into definitions?

        # Containers for creating TopicEstimator instance
        self.param_container = param_container
        self.embeddings_container = embeddings_container
        self.text_container = text_container
        self.estimator = Type[TopicEstimator]

        # Defaults are the initialized state of the modeller instance
        self.dimensionality_reducer_params_to_iterate = dimensionality_reducer_params_to_iterate
        self.clusterer_params_to_iterate = clusterer_params_to_iterate
        self.param_grid = param_grid

        # Set TopicEstimator instance
        self.set_estimator_instance()
        self.set_param_grid(force=False)

        # Results
        self.topic_tables = None
        self.topic_estimates = None
        self.prob_estimates = None
        self.topic_models = None

    def set_estimator_instance(self):
        self.estimator = TopicEstimator(text_container=self.text_container,
                                        embeddings_container=self.embeddings_container,
                                        applied_models=ModelContainer(self.param_container))

    def set_param_grid(self, force=True):
        """
        Sets an iterable paramgrid from the sets of two parameter lists if called. If not forced
        the non-empty param_grid will not be overwridden.
        """
        if force:
            try:
                self.param_grid = list(
                    itertools.product(self.dimensionality_reducer_params_to_iterate, self.clusterer_params_to_iterate))
            except TypeError:
                raise TypeError('Please enter iterables for both param lists.')
        else:
            self.param_grid = self.param_grid or list(
                itertools.product(self.dimensionality_reducer_params_to_iterate, self.clusterer_params_to_iterate))

    def set_umap_reduced_embeddings(self, persist: bool = True):
        reducer_params = self.param_container.dimensionality_reducer_params
        suffix = ''.join(
            [str(_).replace('.', ',') for _ in itertools.chain.from_iterable(reducer_params.items())]) + '.npy'
        reduced_embeddings_path = self.reduced_embeddings_path / suffix
        if not reduced_embeddings_path.exists():
            reducer = self.estimator.dimensionality_reducer
            self.embeddings_container.embeddings = reducer.fit_transform(self.embeddings_container.default_embeddings)
            if persist:
                np.save(reduced_embeddings_path, self.embeddings_container.embeddings)
        else:
            self.embeddings_container.embeddings = np.load(reduced_embeddings_path)

    def reset_embeddings(self):
        self.embeddings_container.embeddings = self.embeddings_container.default_embeddings

    def feed_params_estimate_store(self, reducer_params_to_apply, clusterer_params_to_apply):
        self.param_container.dimensionality_reducer_params = reducer_params_to_apply
        self.param_container.clusterer_params = clusterer_params_to_apply
        self.set_estimator_instance()

        topics, probs, topic_model = self.estimator.estimate_topic_model()
        self.topic_estimates.append(topics)
        self.prob_estimates.append(probs)
        self.topic_models.append(topic_model)

        topic_table = topic_model.get_topic_info()
        topic_table.loc[topic_table.index[0], 'param_set'] = ''.join(
            [str(_) for _ in itertools.chain.from_iterable(reducer_params_to_apply.items())]) + ''.join(
            [str(_) for _ in itertools.chain.from_iterable(clusterer_params_to_apply.items())])
        self.topic_tables.append(topic_table)

    def estimate_and_store_over_param_set(self, optimized: bool = True, persist: bool = True):
        self.topic_tables = []
        self.topic_estimates = []
        self.prob_estimates = []
        self.topic_models = []
        if not optimized:
            for params in tqdm(self.param_grid):
                dimensionality_reducer_params, clusterer_params = params
                self.feed_params_estimate_store(dimensionality_reducer_params, clusterer_params)
            return
        for dimensionality_reducer_params in tqdm(self.dimensionality_reducer_params_to_iterate):
            self.param_container.dimensionality_reducer_params = dimensionality_reducer_params
            self.param_container.dimensionality_reducer_type = UMAP
            self.set_estimator_instance()
            self.set_umap_reduced_embeddings(persist=persist)
            self.param_container.dimensionality_reducer_type = EmptyUMAP
            self.set_estimator_instance()
            for clusterer_params in tqdm(self.clusterer_params_to_iterate):
                self.feed_params_estimate_store(dimensionality_reducer_params, clusterer_params)
        self.reset_embeddings()

    def save_results(self):
        objects_to_save = (self.topic_estimates, self.prob_estimates, self.topic_models)
        with open(self.result_path / 'estimates.pickle', 'wb') as file:
            pickle.dump(objects_to_save, file, protocol=pickle.HIGHEST_PROTOCOL)
        save_xls(self.topic_tables, self.paramgrid_result_path)


def run_topic_modelling():
    corpus = pd.read_csv(project_folders['work'] / 'corpus_sented.csv')
    text_container = TextContainer(corpus_input=corpus, text_varname='sentences')
    embeddings = np.load(project_folders['work'] / 'embedding.npy')
    embeddings_container = EmbeddingsContainer(embeddings=embeddings)
    param_container = ParamContainer()

    umap_params_list = [{'n_neighbors': n, 'n_components': c, 'min_dist': d, 'metric': 'cosine', 'random_state': 42}
                        for n in [5, 20]
                        for c in [2, 3, 5]
                        for d in [.1]]
    hdbscan_params_list = [
        {'min_cluster_size': m, 'metric': 'euclidean', 'cluster_selection_method': 'eom', 'prediction_data': True}
        for m in [5, 15, 30, 50]]

    iterate_topic_estimator = IterateTopicEstimator(param_container=param_container,
                                                    embeddings_container=embeddings_container,
                                                    text_container=text_container,
                                                    dimensionality_reducer_params_to_iterate=umap_params_list,
                                                    clusterer_params_to_iterate=hdbscan_params_list)

    iterate_topic_estimator.set_param_grid()
    iterate_topic_estimator.estimate_and_store_over_param_set(optimized=True, persist=True)
    return iterate_topic_estimator


if __name__ == "__main__":
    iterate_topic_estimator = run_topic_modelling()
    print('finished')
