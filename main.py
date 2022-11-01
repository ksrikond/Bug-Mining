import os
import errno

import pandas as pd
import src.utils.hsd_util as _util

from src.utils.logger_config import api_logger
from src.data.preprocess import Preprocess
from src.model.cluster_k_means import Cluster

cfg = _util.load_configuration('config.yaml')

root_path = os.getcwd()
src_path = os.path.join(root_path, 'src')
datafiles_path = os.path.join(src_path, 'data', 'data_files')
visualization_path = os.path.join(src_path, 'visualization')


class HsdMiningAnalysis:
    """HSD Mining Analysis Application

    Parameters
    ----------
    raw_data: str, default=''
        raw data filename.

    dash: bool, default=False
        True if invoking from dashboard, False otherwise.

    algorithm: {"co-occurrence", "word2vec"}, default="co-occurrence"
        Word embedding algorithm to use. The default algorithm used is "co-occurrence". It will build
        a visual co-occurrence matrix from the given corpus. It's computationally more expensive.
        If "word2vec" is specified as the algorithm, https://arxiv.org/abs/1301.3781 is used.

    ner: {"stanford-ner", "spacy"}, default="stanford-ner"
        Named Entity Recognizer (NER) to use. The default NER used is "stanford-ner".
        NER labels sequences of words in a text which are the names of things, such as person and company names,
        or gene and protein names. HSD Mining Analysis uses NER to extract and filter specific type of entities
        from data (e.g. person, date, time).
        If "spacy" is specified as the NER, https://spacy.io/ is used.
    """

    def __init__(self, raw_data='', algorithm='co-occurrence', ner='stanford-ner', dash=False):
        self.raw_data = raw_data
        self.dash = dash
        self.algorithm = algorithm
        self.ner = ner

    def run_hsd_mining(self):
        """Start HSD Mining and Analysis. This method processes the data from raw_data file using different
        custom methods, strategies written.
        """
        if not self.raw_data:
            raise ValueError(f"Please specify filename to process")

        filename = os.path.join(datafiles_path, 'raw_data', self.raw_data)
        if os.path.exists(filename):
            api_logger.info("Reading HSD data from {0}".format(filename))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        api_logger.info("HSD Mining Application flow started")
        hsd_json = _util.decompress_pickle(filename)
        df = pd.json_normalize(hsd_json)
        df.columns = [elem.split('.')[-1] for elem in df.columns.values]  # removing server.bugeco from column names
        api_logger.info("Columns selected for processing: {0}".format(cfg['main']['columnsToProcess']))

        df = df[cfg['main']['columnsToProcess']]

        preprocess = Preprocess(self.ner, dash=self.dash)
        preprocess.extract_users(df, ['owner', 'notify'])  # notify and owner columns
        preprocess.extract_supporting_features(df, 'submitted_date', 'id')

        #api_logger.info("Parsing comments to extract text from comments thread")
        #df['comments'] = df['comments'].apply(preprocess.parse_comments)

        for col in cfg['main']['columnsToPreprocess']:
            api_logger.info("Cleaning Column: {0}".format(col))
            df[col] = df[col].apply(preprocess.clean_column)

        api_logger.info("Completed data cleaning process")

        if self.ner not in ("stanford-ner", "spacy"):
            raise ValueError(f"Parameter ner must be 'stanford-ner' or 'spacy', "
                             f"got {self.ner} instead.")

        api_logger.info("Extracting named entities list from {0} column using {1}".format(
            ', '.join(['description']),
            self.ner))

        if self.ner == "stanford-ner":
            df[['description']].apply(preprocess.extract_named_entities_using_stanford_ner)

        elif self.ner == "spacy":
            df[['description']].apply(preprocess.extract_named_entities_using_spacy)

        api_logger.info("Named entities from text: {0}".format(preprocess.named_entities))
        preprocess.update_stopwords()

        df['title'] = df['title'].apply(preprocess.increase_column_weight, n_times=3)
        corpus = preprocess.build_corpus(df, cfg['main']['columnsToPreprocess'])

        preprocess.tfidf_vectorizer(corpus, cfg)
        api_logger.info("TF-IDF process complete")

        if self.algorithm not in ("co-occurrence", "word2vec"):
            raise ValueError(f"Algorithm must be 'co-occurrence' or 'word2vec', "
                             f"got {self.algorithm} instead.")

        if self.algorithm == "co-occurrence":
            vocabulary, co_occurrence = preprocess.build_co_occurrence_matrix(corpus,
                                                                              cfg['main']['output']['wordEmbeddingModelFile'][self.algorithm],
                                                                              window=5)
            api_logger.info("There are {0} words in HSD Vocabulary.".format(len(vocabulary)))

            if not self.dash:
                word_clusters, embedding_clusters, labels = _util.produce_word_clusters_from_co_occurrence(
                    co_occurrence,
                    features_file=cfg['main']['output']['featureFile'],
                    n=25,
                    n_similar=10,
                    depth=2)

                filename = os.path.join(datafiles_path, 'co_occurrence_data', cfg['main']['output']['wordClustersFile'][self.algorithm])
                _util.save_word_clusters(filename, labels, word_clusters)

                _util.plot_co_occurrence_matrix(labels, word_clusters, embedding_clusters, filepath=visualization_path)

        elif self.algorithm == "word2vec":
            word2vec_model = preprocess.word2vec_vectorizer(corpus,
                                                            cfg['main']['output']['wordEmbeddingModelFile'][self.algorithm],
                                                            dim=100,
                                                            window=5)
            model_vocab, word_vectors = list(word2vec_model.wv.vocab), word2vec_model.wv.vectors

            api_logger.info("Logging HSD Vocabulary. There are {0} words.".format(len(model_vocab)))
            api_logger.info(model_vocab)

            filename = os.path.join(datafiles_path, 'word2_vec_data', 'word_embeddings.csv')
            pd.DataFrame(word_vectors, index=model_vocab).to_csv(filename)
            api_logger.info("Saved Word2Vec output to {0}".format(filename))

            api_logger.info("Word2Vec process complete")

            if not self.dash:
                word_clusters, embedding_clusters, labels = _util.produce_word_clusters_from_word2vec(word2vec_model,
                                                                                        features_file=cfg['main']['output']['featureFile'],
                                                                                        n=25,
                                                                                        n_similar=10,
                                                                                        depth=2)
                filename = os.path.join(datafiles_path, 'word2_vec_data', cfg['main']['output']['wordClustersFile'][self.algorithm])
                _util.save_word_clusters(filename, labels, word_clusters)

                _util.tsne_plot_similar_words_plotly(labels, word_clusters, embedding_clusters,
                                                     filepath=visualization_path)

                filename = os.path.join(datafiles_path, 'word2_vec_data', 'word_embeddings.csv')
                df = pd.read_csv(filename, index_col=0)
                vocab, vectors = df.index, df.values

                num_clusters = 10  # select initial numbers of clusters
                _cluster = Cluster(n_clusters=num_clusters)
                cluster_labels, cluster_centroids = _cluster.cluster_data(vectors)

                df['cluster-labels'] = cluster_labels
                _util.plot_kmeans_clusters_plotly(df, _cluster.num_clusters, vectors, cluster_centroids,
                                                  filepath=visualization_path)

        api_logger.info("HSD Mining main.py flow complete")


if __name__ == '__main__':
    main = HsdMiningAnalysis(
        raw_data=cfg['main']['rawDataFile'],
        algorithm=cfg['main']['algorithm'],
        ner=cfg['main']['namedEntityRecognizer'],
        dash=False)
    main.run_hsd_mining()
