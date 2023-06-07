from docx import Document
import pandas as pd
import os


class ImportGenCorpusDf():
    def __init__(self, data_folder='.\\data\\'):
        if (data_folder[-1:] != '\\') | (data_folder[-1:] != '//'):
            data_folder = data_folder + '\\'
        self.data_folder: str = data_folder

    def docx_to_string(self):
        filepaths = [self.data_folder + filename for filename in os.listdir(self.data_folder)]
        texts = ['\n'.join([par.text for par in Document(file).paragraphs]).strip() for file in filepaths]
        corpus = pd.DataFrame(
            pd.Series(texts,
                      name='texts')  # indexeket megtartja, de a felrobbantott iterable elemszámával megegyező observationt generál az explode()
            .str.split('\n').explode()).reset_index(names='doc_id')  # az indexek pedig így lesznek a doc_id-k
        return corpus

    def relevant_string_to_corpus(self, corpus):
        # Csak azokat tartjuk meg, amik megfelelnek a [valami]:[szöveg] logikának
        corpus = corpus[corpus.texts.str.count(':') > 0]  # releváns stringek
        corpus = pd.concat(
            [corpus['doc_id'],
             corpus['texts'].str.split(':', n=1)
             .apply(
                 pd.Series)
             .rename({0: 'id', 1: 'text'}, axis=1)],
            axis=1)
        return corpus

    def clean_id(self, corpus):
        corpus = corpus[corpus.id.str.count(' ') < 2]
        # kézzel is szűrhető. itt már nem kell belemenni.
        corpus = corpus[~corpus.id.isin(
            ['[Valaki felsóhajt', '[Kórusban ]', '[női hang]', '[férfi hang]', '[Férfihang]', '[többen összesúgnak',
             '[Valaki suttogva', '[Viccből]'])]
        corpus.id = corpus.id.str.replace('[^a-zA-ZáéíóöőüűÁÉÍÓÖŐÜŰ\s]', '', regex=True)
        corpus = corpus[corpus.id != '']
        return corpus

    def get_corpus(self):
        corpus = self.docx_to_string()
        corpus = self.relevant_string_to_corpus(corpus)
        corpus = self.clean_id(corpus)
        return corpus


if __name__ == '__main__':
    corpus_import = ImportGenCorpusDf()
    corpus = corpus_import.get_corpus()
    corpus.to_csv('test.csv')
