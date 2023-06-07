import pandas as pd
import huspacy
from import_docx import *

nlp = huspacy.load('hu_core_news_lg')
from tqdm.notebook import tqdm

tqdm.pandas()
# stopszószűrés
stopwords = nlp.Defaults.stop_words


class Preprocess():
    # def __init__(self, corpus = pd.DataFrame):
    #      self._corpus: pd.DataFrame() = corpus

    # @property
    # def corpus(self):
    #     return self._corpus
    #
    # @corpus.setter
    # def corpus(self, df):
    #     self._corpus: pd.DataFrame() = df

    def text_cleaning_for_bow(self, corpus):
        corpus.text = corpus.text.str.replace('[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ\s]', '',
                                              regex=True).str.lower().str.strip()
        corpus.text = corpus.text.apply(lambda text: ' '.join([tok for tok in text.split() if tok not in stopwords]))
        return corpus[corpus.text != '']


if __name__ == '__main__':
    corpus_import = ImportGenCorpusDf()
    corpus_df: object = corpus_import.get_corpus()
    corpus_preprocess = Preprocess()
    corpus_df = corpus_preprocess.text_cleaning_for_bow(corpus_df)
    corpus_df.to_csv('test.csv')
