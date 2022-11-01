"""
HSD Mining & Analysis Data Pre-processing

This file contains supporting function required for HSD data pre-processing.
"""
import os
import multiprocessing
import datetime
import spacy
import re
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordNERTagger
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from src.utils.logger_config import api_logger
import src.utils.hsd_util as _util

root_path = os.getcwd()
src_path = os.path.join(root_path, 'src')
datafiles_path = os.path.join(src_path, 'data', 'data_files')
visualization_path = os.path.join(src_path, 'visualization')

stopwords_and_exceptions = os.path.join(datafiles_path, 'raw_data', 'stopwords_and_exceptions.json')


class ComputeLoss(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0

    def on_epoch_end(self, model):
        """Compute and print the loss after each epoch

        Parameters
        ----------
        model: gensim.models.Word2Vec
            Word2Vec model under training
        """
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            api_logger.info("Loss after epoch {0}: {1}".format(self.epoch, loss))
        else:
            api_logger.info("Loss after epoch {0}: {1}".format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


class Preprocess(object):
    """Data Pre-processing for HSD Mining & Analysis

    Parameters
    ----------
    ner: {"stanford-ner", "spacy"}
        Named Entity Recognizer (NER) to use. The default NER used is "stanford-ner".
        NER labels sequences of words in a text which are the names of things, such as person and company names,
        or gene and protein names. HSD Mining Analysis uses NER to extract and filter specific type of entities
        from data (e.g. person, date, time).
        If "spacy" is specified as the NER, https://spacy.io/ is used.

    dash: bool
        True if invoking from dashboard, False otherwise.
    """
    def __init__(self, ner, dash):
        self.dash = dash
        self.stopwords_list = stopwords.words('english')
        self.stopwords_list.extend(brown.words())
        self.named_entities = set()
        self.time_features = []
        self.hsd_ids = []

        if ner == "stanford-ner":
            self.stanford_ner = StanfordNERTagger('stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz',
                                                  'stanford-ner/stanford-ner.jar')
        elif ner == "spacy":
            self.spacy_nlp = spacy.load('en_core_web_sm')
            self.spacy_nlp.max_length = 1000000000

        with open(stopwords_and_exceptions) as json_file:
            data = json.load(json_file)
            self.stopwords_list.extend(data['custom_stopwords'])
            self.exception_list = data['exceptions']
        self.stemmer = WordNetLemmatizer()

    def clean_column(self, data):
        """Data Pre-processing for HSD Mining & Analysis

        Parameters
        ----------
        data: str
            Data to clean from HSD document fields

        Returns
        ----------
        Clean data from every field
        """
        if data:
            api_logger.debug("Logging Data Before Cleaning: {0}".format(data))
            soup = BeautifulSoup(data, features='html.parser')
            data = soup.get_text(separator="\n")
            api_logger.debug("Removing HTML markup tags: {0}".format(data))
            data = re.sub(r'http\S+|\S+/\S+/\S+', ' ', data)
            api_logger.debug("Removing URLs and paths: {0}".format(data))
            data = re.sub(r'\S*@\S*\s?|_{2,}|ww[0-9]{1,2}', ' ', data)
            api_logger.debug("Removing email address, extra underscores, work week numbers(eg. WW01): {0}".format(data))
            # below line removes the linker log lines with patter "[number] AD ADD ..."
            data = re.sub(r'[\[0-9\]]{8,}[\s]+(AD|AK|BL|IV)[\s]+(ADD|DROP).*', ' ', data)
            api_logger.debug(
                "Removing linker log details (e.g. [183119373]  AD ADD  P2C 0x01(CORE)..): {0}".format(data))
            data = re.sub(r'\s\s+', ' ', data)
            api_logger.debug("Removing line breaks and extra spaces: {0}".format(data))
            """
            lowercasing is removed to improve quality of results given by spacy|Stanford NER. Lowercasing is done in
            tokenization and lemmatization step
            """
            # data = data.lower()
            # api_logger.debug("Converting to lowercase: {0}".format(data))
            """
            Below line removes all the pipe separated register address strings
            Note: removing line breaks and extra space step is required for removing register tokens. Hence added here
            as well as in the end
            """
            data = re.sub(r'(.+?\|).*', ' ', data)
            api_logger.debug(
                "Removing register address tokens (e.g. 1429789646 | M | 0x0000feb4d99a): {0}".format(data))
            data = re.sub(r'0x[A-F0-9a-f\S]*', ' ', data)
            api_logger.debug("Removing hexadecimal tokens: {0}".format(data))
            data = re.sub(r'[^\w\s]', ' ', data)
            api_logger.debug("Removing punctuation: {0}".format(data))
            data = re.sub(r'\s+[0-9][A-H0-9a-h_]+\S+', ' ', data)
            api_logger.debug("Removing alphanumeric tokens (e.g. 000000000a60 00000000ffe00365): {0}".format(data))
            data = re.sub(r'\s\s+', ' ', data)
            api_logger.debug("Removing line breaks and extra spaces: {0}".format(data))
            data = ' '.join([i for i in data.split() if not i.strip().isnumeric()])
            api_logger.debug("Removing Numbers: {0}".format(data))
            return data
        return ''

    def extract_users(self, data, columns):
        """Data Pre-processing for HSD Mining & Analysis

        Parameters
        ----------
        data: DataFrame
            DataFrame containing HSD data

        columns: list
            Columns to extract usernames from

        Returns
        ----------
        Iterator of usernames
        """
        users = []
        api_logger.info("Extracting user names from: {0}".format(columns))

        for col in columns:
            if col in data.columns:
                for index, value in data[col].items():
                    users.extend(user.strip() for user in value.split(',') if user.strip() and user not in users)
            else:
                api_logger.warning("Column not present in data, skipping {0}".format(col))

        api_logger.info("Adding usernames to stopwords")
        self.stopwords_list.extend(users)

        return users

    def extract_supporting_features(self, df, column, id_column):
        """Extract Time features and HSD IDs

        Parameters
        ----------
        df: DataFrame
            DataFrame containing HSD data

        column: str
            Time features column

        id_column: str
            HSD ID column
        """
        self.hsd_ids = df[id_column].tolist()  # extract hsd ids
        if column in df.columns:
            api_logger.info("Extracting time and id features using {0} column".format(column))
            date_format = '%Y-%m-%d %H:%M:%S.%f'

            for elem in df[column].values:
                date_obj = datetime.datetime.strptime(elem, date_format)
                self.time_features.append([date_obj.year, date_obj.day, date_obj.month])

            api_logger.info("Finished forming time and id feature matrix: {0}".format(self.time_features))
        else:
            api_logger.warning("Column not present in data, add {0} field to extract time features.".format(column))

    def extract_named_entities_using_stanford_ner(self, data):
        """Extract people names, months, days-of-a-week using Stanford NER

        Parameters
        ----------
        data: pd.Series|str
            Data to extract people names from HSD document fields. Named entities can be extracted from one or more
            columns.
        """
        if isinstance(data, pd.Series):
            for row in data.values:
                tags = self.stanford_ner.tag(row.split())
                self.named_entities.update([tag[0].lower() for tag in tags if tag[1] in ['PERSON', 'DATE', 'TIME'] and
                                            tag[0].lower() not in self.exception_list])

        if isinstance(data, str):
            tags = self.stanford_ner.tag(data.split())
            self.named_entities.update([tag[0].lower() for tag in tags if tag[1] in ['PERSON', 'DATE', 'TIME'] and
                                        tag[0].lower() not in self.exception_list])

    def extract_named_entities_using_spacy(self, data):
        """Extract people names, months, days-of-a-week using spaCy

        Parameters
        ----------
        data: pd.Series|str
            Data to extract people names from HSD document fields. Named entities can be extracted from one or more
            columns.
        """
        if isinstance(data, pd.Series):
            for row in data.values:
                doc = self.spacy_nlp(row)

                for element in doc.ents:
                    entry = str(element.lemma_)
                    if element.label_ in ['PERSON', 'DATE']:
                        # updating named entities set and removing it from data
                        self.named_entities.update([i.lower() for i in entry.split() if i.find('_') is -1 and
                                                    i.lower() not in self.exception_list])
                        # data = data.replace(entry, '')

        elif isinstance(data, str):
            doc = self.spacy_nlp(data)

            for element in doc.ents:
                entry = str(element.lemma_)
                if element.label_ in ['PERSON', 'DATE']:
                    # updating named entities set and removing it from data
                    self.named_entities.update([i.lower() for i in entry.split() if i.find('_') is -1 and
                                                i.lower() not in self.exception_list])
                    # data = data.replace(entry, '')
        # api_logger.info("Named entities from text: {0}".format(self.named_entities))

    # This method is for reference and is not being used anywhere
    def extract_named_entities(self, data):
        """
        Extract people names, months, days-of-a-week using spaCy test method
        """
        if isinstance(data, pd.Series):
            for row in data.values:
                doc = self.spacy_nlp(row)

                for element in doc.ents:
                    entry = str(element.lemma_)
                    if element.label_ in ['TIM', 'DATE', 'MONEY', 'ORG', 'GPE', 'GEO', 'ART', 'EVE', 'NAT', 'PERSON']:
                        # updating named entities set and removing it from data
                        self.named_entities.update([i for i in entry.split() if i not in self.exception_list])
                        # data = data.replace(entry, '')

        elif isinstance(data, str):
            doc = self.spacy_nlp(data)

            for element in doc.ents:
                entry = str(element.lemma_)
                if element.label_ in ['TIM', 'DATE', 'MONEY', 'ORG', 'GPE', 'GEO', 'ART', 'EVE', 'NAT', 'PERSON']:
                    # updating named entities set and removing it from data
                    self.named_entities.update([i for i in entry.split() if i not in self.exception_list])
                    # data = data.replace(entry, '')
        # api_logger.info("Named entities from text: {0}".format(self.named_entities))

    def update_stopwords(self):
        """
        Update stopwords to add named entities
        """
        self.stopwords_list.extend(list(self.named_entities))

    def increase_column_weight(self, data, n_times):
        """Increase column data weight n_times

        Parameters
        ----------
        data: str
            Data from a particular column

        n_times: int
            Number of times weight is increased

        Returns
        ----------
        Data after increasing the weight
        """
        data = (data + ' ') * n_times
        return data

    def parse_comments(self, data):
        """Parse comments to get text

        Parameters
        ----------
        data: str
            Data from a particular column

        Returns
        ----------
        Extracted text from comments column

        Example
        ----------
        HSD data has 'comments' column which fetches comments in a specific format
        For example,
        ++++1465384938 jbstein
        comment text here

        ++++2266238612 adanabal
        comment text here
        """
        if data.strip():
            comment_list = list(filter(bool, data.splitlines()))
            comment_dict, comm_key = {}, ''

            for line in comment_list:
                if line[0:4] == '++++':
                    comm_key = line[4:]
                else:
                    if comm_key not in comment_dict:
                        comment_dict[comm_key] = line
                    else:
                        comment_dict[comm_key] += line

            return '\n'.join(comment_dict.values())

        return None

    #  Below method was created for test. Not in use
    def tokenize_and_lemmatize_test(self, data):
        """
        Tokenize and Lemmatize data test method
        """
        api_logger.debug("Tokenizing data to return a list of tokens")

        tokens = [word.lower() for word in word_tokenize(data) if
                  (len(word) > 1) and (word.lower() not in self.stopwords_list)]
        api_logger.debug("Lemmatization starts using Word Net Lemmatizer")

        return [self.stemmer.lemmatize(token) for token in tokens]

    def tokenize_and_lemmatize(self, col_data):
        """Convert a paragraph into list of tokens and perform lemmatization

        Parameters
        ----------
        col_data: pd.Series
            Data from a particular column

        Returns
        ----------
            Iterator of word tokens
        """
        api_logger.debug("Tokenizing and Lemmatizing data using Word Net Lemmatizer")

        tokens_list = []
        for data in col_data:
            tokens = []
            for word in word_tokenize(data):
                word = self.stemmer.lemmatize(word.lower())
                if len(word) > 1 and word not in self.stopwords_list:
                    tokens.append(word)
            tokens_list.append(tokens)
        return tokens_list

    #  Below method was created for test. Not in use
    def build_corpus_test(self, data, columns):
        """
        Build HSD Corpus test method
        """
        api_logger.info("Building corpus from {0} column".format(', '.join(columns)))

        corpus = data[columns[0]].apply(self.tokenize_and_lemmatize).values.tolist()
        for i in range(1, len(columns)):
            # print(columns[i])
            temp_list = data[columns[i]].apply(self.tokenize_and_lemmatize).values.tolist()
            for x, y in zip(corpus, temp_list):
                x.extend(y)
            # print("done")
        return corpus

    def build_corpus(self, data, columns):
        """Convert a paragraph into list of tokens and perform lemmatization

        Parameters
        ----------
        data: DataFrame
            DataFrame containing HSD data

        columns: list
            Columns to consider while building corpus

        Returns
        ----------
        Corpus of HSD Documents
        """
        api_logger.info("Building corpus from {0} column".format(', '.join(columns)))

        corpus = self.tokenize_and_lemmatize(data[columns[0]].values)
        for i in range(1, len(columns)):
            temp_list = self.tokenize_and_lemmatize(data[columns[i]].values)
            for j in range(len(corpus)):
                corpus[j].extend(temp_list[j])

        return corpus

    def tfidf_vectorizer(self, corpus, cfg):
        """Perform vectorization on HSD data using Term Frequency - Inverse Document Frequency(TF-IDF)
        Save output of vectorization to csv file.

        Parameters
        ----------
        corpus: list
            corpus containing HSD data

        cfg: dict
            HSD Mining configuration properties
        """
        api_logger.info("Computing Term Frequency-Inverse Document Frequency")
        # Prepare an iterator of documents
        documents = []
        for doc in corpus:
            documents.append(' '.join(doc))

        tf_idf_vectorizer = TfidfVectorizer(max_features=20000, norm=None)

        tf_idf = tf_idf_vectorizer.fit_transform(documents)
        tf_idf_array = tf_idf.toarray()

        tf_idf_array = np.hstack(
            (tf_idf_array, np.array(self.time_features)))  # concatenating time features to if-idf features
        column_names = tf_idf_vectorizer.get_feature_names() + ['Year', 'Day', 'Month']
        df_tfidf = pd.DataFrame(tf_idf_array, columns=column_names)
        df_tfidf['hsd_id'] = self.hsd_ids

        filename = os.path.join(datafiles_path, 'tf_idf_data', cfg['main']['output']['tfIdfFile'])
        api_logger.info("Saving the TF-IDF vector to {0}".format(filename))
        df_tfidf.to_csv(filename, index=False)

        df_freq = _util.get_features_list(df_tfidf)
        filename = os.path.join(datafiles_path, 'tf_idf_data', cfg['main']['output']['featureFile'])
        api_logger.info("Saving TF-IDF features list to {0}".format(filename))
        df_freq.to_csv(filename, header=['tf-idf'], index_label='features')

        if not self.dash:
            _util.generate_wordcloud_visualization(df_freq)
        return

    def word2vec_vectorizer(self, corpus, filename, dim=100, window=5):
        """Perform Vectorization using Word2Vec algorithm(a simple Neural Network model)

        Parameters
        ----------
        corpus: list
            corpus containing HSD data

        filename: str
            Filename to save the trained model to

        dim: int
            Vector dimensions

        window: int
            Size of the sliding window

        Returns
        ----------
        A trained Word2Vec model on HSD data
        """
        api_logger.info("Performing word2vector vectorization")

        word2vec_model = Word2Vec(corpus,
                                  size=dim,
                                  window=window,
                                  min_count=1,
                                  max_final_vocab=500000,
                                  compute_loss=True,
                                  # iter=10, # for testing remove after use
                                  iter=400,
                                  # hs=1,
                                  # sg=0,
                                  # negative=0,
                                  callbacks=[ComputeLoss()],
                                  workers=multiprocessing.cpu_count()
                                  )

        _util.save_word2vec_model(word2vec_model, filename)

        return word2vec_model

    def build_co_occurrence_matrix(self, corpus, filename, window=5):
        """Build visual co-occurrence matrix

        Parameters
        ----------
        corpus: list
            corpus containing HSD data

        filename: str
            Filename to save the matrix to

        window: int
            Size of the sliding window

        Returns
        ----------
        Vocabulary and co-occurrence matrix
        """
        vocabulary = _util.build_vocabulary(corpus)

        api_logger.info("Building co-occurrence matrix")
        co_occurrence = {i: Counter({}) for i in vocabulary}

        for document in corpus:
            for i in range(len(document)):
                if i < window:
                    c = Counter(document[0:i + window + 1])
                    del c[document[i]]
                    co_occurrence[document[i]] = co_occurrence[document[i]] + c
                elif i > len(document) - (window + 1):
                    c = Counter(document[i - window::])
                    del c[document[i]]
                    co_occurrence[document[i]] = co_occurrence[document[i]] + c
                else:
                    c = Counter(document[i - window:i + window + 1])
                    del c[document[i]]
                    co_occurrence[document[i]] = co_occurrence[document[i]] + c

        api_logger.info("Completed building co-occurrence matrix")
        _util.save_co_occurrence_matrix(co_occurrence, filename)

        return vocabulary, co_occurrence

    """ This method is an implementation of Word2Vec from scratch using TensorFlow. Not in use
    def word2vec_vectorizer(self, vocabulary, data):
        api_logger.info("Performing word2Vector vectorization")
        word2int, int2word, vocab_size = {}, {}, len(vocabulary)

        for i, word in enumerate(vocabulary):
            word2int[word] = i
            int2word[i] = word
        sentences = []
        for index, value in data.items():
            sentences.append(value)

        WINDOW_SIZE = 2

        data = []
        for sentence in sentences:
            for index, word in enumerate(sentence):
                for nb_word in sentence[max(index - WINDOW_SIZE, 0): min(index + WINDOW_SIZE, len(sentence)) + 1]:
                    if nb_word != word and (word in word2int) and (nb_word in word2int):
                        data.append([word, nb_word])

        x_train = []
        y_train = []
        for element in data:
            x_train.append(self.to_one_hot(word2int[element[0]], vocab_size))
            y_train.append(self.to_one_hot(word2int[element[1]], vocab_size))
        # converting to numpy array
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        # making placeholders for training variables
        x = tf.placeholder(tf.float32, shape=(None, vocab_size))
        y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

        EMBEDDING_DIM = 10
        # initializing weights and bias
        W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
        b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))
        hidden_representation = tf.add(tf.matmul(x, W1), b1)

        W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
        b2 = tf.Variable(tf.random_normal([vocab_size]))
        prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))

        tf_session = tf.Session()
        init = tf.global_variables_initializer()
        tf_session.run(init)

        loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

        training_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        epochs = 50000

        for _ in range(epochs):
            tf_session.run(training_step, feed_dict={x: x_train, y_label: y_train})

        vectors = tf_session.run(W1 + b1)
        api_logger.info("Word2Vec vectorization process complete")
        visualize_word_vectors(vectors, vocabulary, word2int)
        return vectors
    """

    def to_one_hot(self, data_point_index, vocab_size):
        """
        TensorFlow implementation of Word2Vec requires one hot encoding. Not in use.
        """
        vec = np.zeros(vocab_size)
        vec[data_point_index] = 1

        return vec

    def visualize_column(self, data):
        """
        WordCloud visualization for column text. Test method, Not in use.
        """
        wordcloud = WordCloud(collocations=False,
                              width=1600, height=800,
                              background_color='white',
                              stopwords=self.stopwords_list,
                              max_words=150,
                              random_state=42
                              ).generate(' '.join(data))

        api_logger.info("Saving visualization after data preprocessing to src/visualization {0}".format(wordcloud))
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(9, 8))
        fig = plt.figure()
        plt.imshow(wordcloud)
        plt.axis('off')
        filename = os.path.join(visualization_path, 'wordcloud_plot_' + data.name)
        fig.savefig(filename)
