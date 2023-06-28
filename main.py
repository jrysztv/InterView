from pathlib import Path

from text_processing.embedding import SentenceEmbedder
from text_processing.import_docx import DocumentToDataFrame
from text_processing.sentencize import sentencize_dataframe

if not Path('work-files/corpus.csv').is_file():
    print('Reading corpus')
    corpus_generator = DocumentToDataFrame()
if not Path('work-files/corpus_sented.csv').is_file():
    print('Sentencizing')
    df_sented = sentencize_dataframe()
if not Path('work-files/embedding.npy').is_file():
    print('Embedding')
    embedder = SentenceEmbedder()
    embeddings = embedder.retrieve_embeddings(load_from_file=False)
