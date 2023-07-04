from pathlib import Path

from text_processing.embedding import SentenceEmbedder
# from text_processing import embedding.SentenceEmbedder as valamilofasz
from text_processing.import_docx import DocumentToDataFrame
from text_processing.sentencize import sentencize_dataframe

if not Path('work_files/corpus.csv').is_file():
    print('Reading corpus')
    corpus_generator = DocumentToDataFrame()
if not Path('work_files/corpus_sented.csv').is_file():
    print('Sentencizing')
    df_sented = sentencize_dataframe()
if not Path('work_files/embedding.npy').is_file():
    print('Embedding')
    embedder = SentenceEmbedder()
    embedder.retrieve_embeddings(
        text_varname='sentences',
        csv_file_path=Path('work_files/corpus_sented.csv'),
        embedding_file_path=Path('work_files/embedding_sentences.npy'),
        load_from_file=False
    )
    embedder.retrieve_embeddings(
        text_varname='text',
        csv_file_path=Path('work_files/corpus.csv'),
        embedding_file_path=Path('work_files/embedding_text.npy'),
        load_from_file=False
    )
