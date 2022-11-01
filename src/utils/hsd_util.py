"""
HSD Mining & Analysis Util

This file contains supporting utility function required for HSD Mining Process.
"""
import os
import bz2
import plotly
import pickle
import itertools
import yaml

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go

import numpy as np
import pandas as pd

from matplotlib import cm
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from collections import deque
from gensim.models import KeyedVectors
# from celluloid import Camera

from src.utils.logger_config import api_logger


root_path = os.getcwd()
src_path = os.path.join(root_path, 'src')
datafiles_path = os.path.join(src_path, 'data', 'data_files')
visualization_path = os.path.join(src_path, 'visualization')
result_dir = os.path.join(src_path, 'results')
model_dir = os.path.join(src_path, 'model')

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
          '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
          '#ffffff', '#000000']


def load_configuration(filepath):
    """Load yaml based configuration file and return config json object

    Parameters
    ----------
    filepath: str
        Path to config.yaml file

    Returns
    ----------
    cfg : Configuration properties in the form of json
    """
    with open(filepath, 'r') as config_file:
        cfg = yaml.safe_load(config_file)
    return cfg


def decompress_pickle(filepath):
    """Decompress pickle data file

    Parameters
    ----------
    filepath: str
        Path to pickle data file

    Returns
    ----------
    hsd_data : HSD data in the json form
    """
    api_logger.info("Decompressing {0} file to json".format(filepath))

    pickle_file = bz2.BZ2File(filepath, 'rb')
    hsd_data = pickle.load(pickle_file)
    pickle_file.close()
    return hsd_data


def append_before(param_list, prefix):
    """Append prefix before every element of param_list and return the comma separated string

    Parameters
    ----------
    param_list: list
        Iterator of strings

    prefix: str
        Prefix to append to elements of param_list

    Returns
    ----------
    str : String having comma separated parameters with prefix appended
    """
    result = ''
    for i in range(len(param_list)):
        result += prefix + param_list[i] + ","
    return result[:-1]


def save_word_clusters_with_ids(filepath, labels, word_clusters, ids):
    """Save 'label: Similar word cluster: HSD IDs' to file

    Parameters
    ----------
    filepath: str
        filename to save

    labels: list
        Iterator of label strings

    word_clusters: list
        Iterator of iterator having strings

    ids: list
        Iterator of HSD IDs corresponding to label
    """
    api_logger.info("Writing Clusters of similar word to {0}".format(filepath))

    with open(filepath, 'a') as file:
        file.write("Word: Cluster of similar words\n")

        for i in range(len(labels)):
            file.write(
                labels[i] + ': ' + ', '.join(word_clusters[i]) + ' HSD IDs: ' + ', '.join(map(str, ids[i])) + '\n')

        file.write('\n')


def save_word_clusters(filepath, labels, word_clusters):
    """Save 'label: Similar word cluster' to file

    Parameters
    ----------
     filepath: str
        filename to save

    labels: list
        Iterator of label strings

    word_clusters: list
        Iterator of iterator having strings
    """
    api_logger.info("Writing Clusters of similar word to {0}".format(filepath))

    with open(filepath, 'a') as file:
        file.write("Word: Cluster of similar words\n")

        for i in range(len(labels)):
            file.write(labels[i] + ': ' + ', '.join(word_clusters[i]) + '\n')

        file.write('\n')


def produce_word_clusters_from_word2vec(model, n=25, n_similar=10, depth=2, features_file=None, top_vocab_words=None):
    """Build word clusters based on word2vec model

    Parameters
    ----------
    model: gensim.models.Word2Vec
        Object of gensim.models.Word2Vec. A trained Word2Vec model.

    n: int, default=25
        Number of words to generate word clusters for

    n_similar: int, default=10
        Number of context words for each word in n

    depth: int, default=2
        Depth of the tree structure

    features_file: str, default=None
        Filename to select top word features from (csv file). An alternative to top_vocab_words

    top_vocab_words: list, default=None
        Iterator of top word features. An alternative to features_file

    Returns
    ----------
    labels : Iterator of word features
    word_clusters : Iterator of iterator having word cluster corresponding to labels
    embedding_clusters : Iterator of iterator having word embedding cluster corresponding words in word_clusters

    Example
    ----------
    If top_vocab_words are ['bios', 'pcode', 'snc'],
    then for n=3, n_similar=2, depth=0 result would be,
    labels = ['bios', 'pcode', 'snc']
    word_clusters = [['icx', 'crif'], ['mc', 'bios'], ['cxl', 'glc']]

    When depth grows, the cluster of labels and related words grows.

    Refer src/references/word-context-explanation.jpeg for explanation of how relations are built based on word context.
    """
    vocabulary = list(model.wv.vocab)  # access model vocabulary
    if top_vocab_words is None:
        top_vocab_words = get_n_words(n, features_file)

    if len(top_vocab_words) != n:
        top_vocab_words = top_vocab_words[:n]  # this will never throw index out of bound even if n>len

    top_vocab_words = [word for word in top_vocab_words if word in vocabulary]  # allowing only word2vec vocab words
    api_logger.info("Logging word selected for clusters: {0}".format(top_vocab_words))

    labels, embedding_clusters, word_clusters = [], [], []

    for word in top_vocab_words:
        queue, level = deque(), 0

        if word not in labels:  # avoid repeating same words
            queue.append(word)

        upper_bound = n_similar ** depth
        while queue and level <= upper_bound:  # collecting only upto 2 levels, replace 2 with depth
            words, embeddings = [], []
            current = queue.popleft()  # returns first word from the queue
            labels.append(current)  # saving labels for plotting
            level += n_similar

            for similar_word, _ in model.most_similar(current, topn=n_similar):
                words.append(similar_word)
                embeddings.append(model[similar_word])

            queue.extend([elem for elem in words if elem not in labels and elem not in queue])
            word_clusters.append(words)
            embedding_clusters.append(embeddings)

    return word_clusters, embedding_clusters, labels


def produce_word_clusters_from_co_occurrence(co_occurrence, n=25, n_similar=10, depth=2, features_file=None, top_vocab_words=None):
    """Build word clusters based on visual co-occurrence matrix

    Parameters
    ----------
    co_occurrence: dict
        A visual co-occurrence matrix object

    n: int, default=25
        Number of words to generate word clusters for

    n_similar: int, default=10
        Number of context words for each word in n

    depth: int, default=2
        Depth of the tree structure

    features_file: str, default=None
        Filename to select top word features from (csv file). An alternative to top_vocab_words

    top_vocab_words: list, default=None
        Iterator of top word features. An alternative to features_file

    Returns
    ----------
    labels : Iterator of word features
    word_clusters : Iterator of iterator having word cluster corresponding to labels
    embedding_clusters : Iterator of iterator having word embedding cluster corresponding words in word_clusters

    Examples
    ----------
    If top_vocab_words are ['bios', 'pcode', 'snc'],
    then for n=3, n_similar=2, depth=0 result would be,
    labels = ['bios', 'pcode', 'snc']
    word_clusters = [['icx', 'crif'], ['mc', 'bios'], ['cxl', 'glc']]

    When depth grows, the cluster of labels and related words grows.

    Refer src/references/word-context-explanation.jpeg for explanation of how relations are built based on word context.
    """
    vocabulary = list(co_occurrence.keys())  # access vocabulary from dict
    if top_vocab_words is None:
        top_vocab_words = get_n_words(n, features_file)

    if len(top_vocab_words) != n:
        top_vocab_words = top_vocab_words[:n]

    top_vocab_words = [word for word in top_vocab_words if word in vocabulary]  # allowing only word2vec vocab words
    api_logger.info("Logging word selected for clusters: {0}".format(top_vocab_words))

    labels, embedding_clusters, word_clusters = [], [], []

    for word in top_vocab_words:
        queue, level = deque(), 0

        if word not in labels:  # avoid repeating same words
            queue.append(word)

        upper_bound = n_similar ** depth
        while queue and level <= upper_bound:  # collecting only upto 2 levels, replace 2 with depth
            words, embeddings = [], []
            current = queue.popleft()  # returns first word from the queue
            labels.append(current)  # saving labels for plotting
            level += n_similar

            for similar_word, embedding in co_occurrence[current].most_common(n_similar):
                words.append(similar_word)
                embeddings.append(embedding)

            queue.extend([elem for elem in words if elem not in labels and elem not in queue])
            word_clusters.append(words)
            embedding_clusters.append(embeddings)

    return word_clusters, embedding_clusters, labels


def get_n_words(n, filename):
    """Get top n word features from the feature file

    Parameters
    ----------
    n: int
        Number of word features

    filename: str
        Filename to select top recurrent features from(tf_idf_data/features_tf_idf*)

    Returns
    ----------
    Iterator of word features
    """
    filename = os.path.join(datafiles_path, 'tf_idf_data', filename)
    df = pd.read_csv(filename)

    vocab_words = []
    for word in df['features'].iloc[:n].values.tolist():
        vocab_words.append(word)

    return vocab_words


def save_word2vec_model(model, filename):
    """Save Word2Vec model into a binary format

    Parameters
    ----------
    model: gensim.models.Word2Vec
        A trained Word2Vec model.

    filename: str
        pre-trained model filename
    """
    filename = os.path.join(model_dir, filename)

    api_logger.info("Saving Word2Vec model in binary format in {0}".format(filename))
    model.wv.save_word2vec_format(filename, binary=True)


def save_co_occurrence_matrix(co_occurrence, filename):
    """Save Word2Vec model into a binary format

    Parameters
    ----------
    co_occurrence: dict
        A trained Word2Vec model.

    filename: str
        Filename to save HSD visual co-occurrence matrix
    """
    filename = os.path.join(model_dir, filename)

    api_logger.info("Saving co_occurrence matrix in {0}".format(filename))
    with open(filename, 'wb') as file:
        pickle.dump(co_occurrence, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_word2vec_model(filepath):
    """Load trained Word2Vec model from file

    Parameters
    ----------
    filepath: str
        Filename to load Word2Vec model from

    Returns
    ----------
    Pre-trained Word2Vec(gensim.models.Word2Vec) model
    """
    if os.path.exists(filepath):
        api_logger.info("Loading pretrained HSD word2vec model")
        word2vec_model = KeyedVectors.load_word2vec_format(filepath, binary=True)
        return word2vec_model
    else:
        api_logger.error("Word2Vec model not found at {0}".format(filepath))


def load_co_occurrence_matrix(filepath):
    """Load pre-computed visual co-occurrence matrix

    Parameters
    ----------
    filepath: str
        Filename to load co-occurrence matrix from

    Returns
    ----------
    dict: pre-computed visual co-occurrence matrix
    """
    if os.path.exists(filepath):
        api_logger.info("Loading precomputed HSD visual co-occurrence matrix")
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data
    else:
        api_logger.error("Visual co-occurrence file not found at {0}".format(filepath))


def build_vocabulary(corpus):
    """Build Vocabulary of unique words from data

    Parameters
    ----------
    corpus: list
        Corpus is collection of raw documents in a clean form

    Returns
    ----------
    Iterator of unique HSD words
    """
    api_logger.info("Generating vocabulary for co-occurrence matrix")
    vocabulary = list(set(itertools.chain.from_iterable(corpus)))
    return vocabulary


def generate_wordcloud_visualization(frequencies):
    """Produce WordCloud visualization from words and their frequencies

    Parameters
    ----------
    frequencies: DataFrame
        A DataFrame having words and their frequencies across all the documents
    """
    api_logger.info("Visualisation of words based on TF-IDF")

    wordcloud = WordCloud(collocations=False,
                          width=1600, height=1600,
                          background_color='white',
                          max_words=500,
                          random_state=42
                          )

    wordcloud.generate_from_frequencies(frequencies)
    plt.figure(figsize=(16, 16))
    plt.imshow(wordcloud)
    plt.axis('off')

    filename = os.path.join(visualization_path, 'tfidf_wordcloud_plots', 'plot_tfidf_wordcloud')
    api_logger.info("Saving TF-IDF visualization to {0}".format(filename))
    plt.savefig(filename)


def get_features_list(df):
    """Get word features and their frequencies across all the documents. Frequencies are normalized across
    all the documents.

    Parameters
    ----------
    df: DataFrame
        A DataFrame storing TF-IDF values word features for all the document.

    Returns
    ----------
    DataFrame: word features and their frequencies
    """
    df_frequencies = df.iloc[:, :-4].T.sum(axis=1).sort_values(ascending=False)
    df_frequencies = normalize_frequencies(df_frequencies)
    return df_frequencies


def normalize_frequencies(df_frequencies):
    """Normalize TF-IDF values using L2 norm.
    Understand L2 norm - https://machinelearningmastery.com/vector-norms-machine-learning/

    Parameters
    ----------
    df_frequencies: DataFrame
        A DataFrame storing word features and their frequencies

    Returns
    ----------
    DataFrame: word features and their frequencies(normalized)
    """
    api_logger.info("Normalizing frequencies across feature space")

    norm_factor = np.sqrt(np.sum(df_frequencies.values ** 2))  # l2 normalization
    df_frequencies.values[:] = df_frequencies.values / norm_factor

    return df_frequencies


# This method is for reference and is not being used anywhere
def visualize_word_vectors(word2vec_model, filepath):
    """
    Visualize word vectors test method
    """
    api_logger.info("Reducing dimensions from {0} to {1} using tSNE".format(word2vec_model.vector_size, 2))
    labels, tokens = [], []

    for word in word2vec_model.wv.vocab:
        labels.append(word)
        tokens.append(word2vec_model[word])

    tsne_model = TSNE(perplexity=30.0, n_components=2, random_state=30, init='pca', n_iter=3000)
    vectors_new = tsne_model.fit_transform(tokens)
    # vectors = normalize(vectors_new)
    api_logger.info("Saving visualization for word vectors in visualization/")
    plt.figure(figsize=(10, 10))
    for i in range(len(vectors_new)):
        plt.scatter(vectors_new[i][0], vectors_new[i][1])  # sending x and y coordinates
        plt.annotate(labels[i],
                     xy=(vectors_new[i][0], vectors_new[i][1]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    filename = os.path.join(filepath, 'plot_word2vec')
    plt.savefig(filename)


def plot_co_occurrence_matrix(labels, word_clusters, embedding_clusters, filepath=None):
    """Plot word clusters built based on co-occurrence.
    This method builds a network graph from label-word cluster pairs. This network graph is then converted
    into a plotly graph.

    Parameters
    ----------
    labels: list
        Iterator of word features

    word_clusters: list
        Iterator of iterator having word cluster corresponding to labels

    embedding_clusters: list
        Iterator of iterator having word embedding cluster corresponding words in word_clusters

    filepath: str, default=None
        Filename to save the plot. None if being called from the dashboard.

    Returns
    ----------
    plotly.graph_objects.Figure object with word cluster properties set
    """
    api_logger.info("Plotting co-occurrence matrix in network graph form using plotly")

    graph = nx.Graph()
    for i in range(len(labels)):
        graph.add_node(labels[i], size=10)
        for j in range(len(word_clusters[i])):
            graph.add_node(word_clusters[i][j], size=embedding_clusters[i][j])
            graph.add_edge(labels[i], word_clusters[i][j], weight=embedding_clusters[i][j])
    pos_ = nx.spring_layout(graph)

    # loading colors for plot
    new_cmap = rand_cmap(len(labels), type='bright', first_color_black=True, last_color_black=False, verbose=False)

    node_traces = []
    for index, (label, words) in enumerate(zip(labels, word_clusters)):
        x, y = pos_[label]
        node_trace = go.Scatter(x=[x],
                                y=[y],
                                name=label,
                                mode='lines+markers+text',
                                marker=dict(size=[10], color='rgba' + str(new_cmap(index))),
                                text=['<b>' + label + '<b>'],
                                hoverinfo='text',
                                textposition='top center')
        for word in words:
            x, y = pos_[word]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['marker']['size'] += tuple([10])
            node_trace['text'] += tuple(['<b>' + word + '<b>'])
        node_traces.append(node_trace)

    layout = go.Layout(
        title='Plot of similar words based on co-occurrence matrix',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        hovermode="closest",
        xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'automargin': True},
        yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'automargin': True}
    )

    fig = go.Figure(layout=layout)
    for trace in node_traces:
        fig.add_trace(trace)

    if filepath is None:
        return fig

    filename = os.path.join(filepath, 'co_occurrence_plots', 'plotly_plot_co_occurrence.html')
    api_logger.info("Saving co-occurrence network plot to {0}".format(filename))
    plotly.offline.plot(fig, filename=filename)


# This method is for reference and is not being used anywhere
def tmp_plot_co_occurrence_matrix(labels, word_clusters, embedding_clusters, filepath=None):
    """
    Visualize word vectors(co-occurrence) test method
    This code connects each label word to all the other words
    """
    api_logger.info("Plotting co-occurrence matrix in network graph form using plotly")

    graph = nx.Graph()
    for i in range(len(labels)):
        graph.add_node(labels[i], size=10)
        for j in range(len(word_clusters[i])):
            graph.add_node(word_clusters[i][j], size=embedding_clusters[i][j])
            graph.add_edge(labels[i], word_clusters[i][j], weight=embedding_clusters[i][j])
    pos_ = nx.spring_layout(graph)

    edge_trace = []
    for edge in graph.edges():
        node_1 = edge[0]
        node_2 = edge[1]
        x0, y0 = pos_[node_1]
        x1, y1 = pos_[node_2]
        text = node_1 + '-->' + node_2 + ': ' + str(graph.edges()[edge]['weight'])
        trace = make_edge([x0, x1, None], [y0, y1, None], text)
        edge_trace.append(trace)

    node_trace = go.Scatter(x=[],
                            y=[],
                            text=[],
                            textposition='middle center',
                            mode='markers+text',
                            hoverinfo='text',
                            marker=dict(color=[],
                                        size=[],
                                        line=None))
    for node in graph.nodes():
        x, y = pos_[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += tuple(['cornflowerblue'])
        node_trace['marker']['size'] += tuple([10])
        # node_trace['marker']['size'] += tuple([graph.nodes()[node]['size']])
        node_trace['text'] += tuple(['<b>' + node + '<br>'])

    layout = go.Layout(
        title='Plot of similar words based on co-occurrence matrix',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        hovermode="closest",
        xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'automargin': True},
        yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'automargin': True}
    )

    fig = go.Figure(layout=layout)
    for trace in edge_trace:
        fig.add_trace(trace)
    fig.add_trace(node_trace)

    if filepath is None:
        return fig

    filename = os.path.join(filepath, 'co_occurrence_plots', 'plotly_plot_co_occurrence.html')
    api_logger.info("Saving co-occurrence network plot to {0}".format(filename))
    plotly.offline.plot(fig, filename=filename)


def make_edge(x, y, text):
    """Get word features and their frequencies across all the documents. Frequencies are normalized across
    all the documents.

    Parameters
    ----------
    x: list
        x co-ordinates of source node
    y: list
        y co-ordinates of source node
    text: str
        text to be shown on scatter lines

    Returns
    ----------
    plotly.graph_objects.Scatter object
    """
    return go.Scatter(x=x,
                      y=y,
                      mode='lines',
                      hoverinfo='text',
                      text=text,
                      line=dict(color='cornflowerblue')
                      )


def tsne_plot_similar_words_plotly(labels, word_clusters, embedding_clusters, filepath=None):
    """Plot word clusters built based on Word2Vector embeddings.
    This method reduced vectors dimensions to 2-D using Principle Component Analysis(PCA). These 2-D embeddings are then
    converted into a plotly graph.

    Parameters
    ----------
    labels: list
        Iterator of word features

    word_clusters: list
        Iterator of iterator having word cluster corresponding to labels

    embedding_clusters: list
        Iterator of iterator having word embedding cluster corresponding words in word_clusters

    filepath: str, default=None
        Filename to save the plot. None if being called from the dashboard.

    Returns
    ----------
    plotly.graph_objects.Figure object with word cluster properties set
    """
    api_logger.info("Plotting Word2vec similar words using plotly")

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    model = PCA(n_components=2)
    vectors = model.fit_transform(embedding_clusters.reshape((n * m, k)))
    embedding_en_2d = np.array(vectors)
    embedding_en_2d = embedding_en_2d.reshape(n, m, 2)

    new_cmap = rand_cmap(len(labels), type='bright', first_color_black=True, last_color_black=False, verbose=False)

    fig = go.Figure(layout=dict(title='Plot of similar words based on Word2Vec'))
    for index, (label, embeddings, words) in enumerate(zip(labels, embedding_en_2d, word_clusters)):
        x = embeddings[:, 0]
        y = embeddings[:, 1]

        fig.add_trace(go.Scatter(x=x, y=y,
                                 name=label,
                                 mode='lines+markers',
                                 marker=dict(size=10, color='rgba' + str(new_cmap(index))),
                                 text=words[:], textposition='top center'))

        # for i, word in enumerate(words):
        #     fig.add_annotation(x=x[i], y=y[i],
        #                        text=word,
        #                        font=dict(family='Arial', size=15),
        #                        exclude_empty_subplots=True,
        #                        showarrow=True,
        #                        visible=True,
        #                        clicktoshow='onoff')

    if filepath is None:
        return fig

    filename = os.path.join(filepath, 'word2vec_cluster_plots', 'plotly_plot_similar_words.html')
    api_logger.info("Saving Word2vec similar words plot to {0}".format(filename))
    plotly.offline.plot(fig, filename=filename)


# This method is for reference and is not being used anywhere
def tsne_plot_similar_words(labels, word_clusters, embedding_clusters, filepath=visualization_path):
    """
    Visualize word vectors(Word2Vec embeddings) test method
    This method used TSNE for dimensionality reduction and matplotlib visualization
    Read TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    api_logger.info("Saving Word2vec similar words plot to src/visualization")

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3000, random_state=32)
    vectors = tsne_model.fit_transform(embedding_clusters.reshape(n * m, k))
    vectors_normalized = normalize(vectors)
    embedding_en_2d = np.array(vectors_normalized)
    embedding_en_2d = embedding_en_2d.reshape(n, m, 2)
    plt.figure(figsize=(16, 16))
    cmap = get_cmap(len(labels))
    for i, (label, embeddings, words) in enumerate(zip(labels, embedding_en_2d, word_clusters)):

        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.title('Clustering words similar to 10 Random vocabulary words')
        plt.grid(True)
        plt.plot(x, y, c=[cmap(i)], alpha=0.5, zorder=1)
        plt.scatter(x, y, c=[cmap(i)], alpha=0.9, label=label, zorder=1)
        plt.legend(loc=4)
        for index, word in enumerate(words):
            plt.annotate(word,
                         alpha=0.9,
                         xy=(x[index], y[index]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom',
                         size=8)
            plt.pause(0.5)
        plt.pause(2.0)

    filename = os.path.join(filepath, 'word2vec-cluster-plots', 'cluster_plot_10')
    plt.savefig(filename)
    # plt.clf()
    # if filename:
    #     plt.savefig(filename)
    # plt.show()


# This method is for reference and is not being used anywhere
def plot_kmeans_clusters(df, num_cluster, word_vectors, cluster_centroids, filepath):
    """
    Visualize word vectors(Word2Vec embeddings) after KMeans clustering
    This method used TSNE for dimensionality reduction and matplotlib visualization
    Read TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Note: Clustering is only used to cross validate Word2Vec word-embeddings. It's not being used in dashboard
    """
    word_vectors = np.array(word_vectors)
    cluster_centroids = np.array(cluster_centroids)
    temp_arr = np.vstack((word_vectors, cluster_centroids))  # combining two for dimensionality reduction

    tsne_model = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3000, random_state=32)
    vectors_2d = tsne_model.fit_transform(temp_arr)
    # centroids_2d = tsne_model.fit_transform(cluster_centroids)
    vectors_2d = np.array(vectors_2d)
    word_vectors_2d = vectors_2d[:len(word_vectors), ]
    centroids_2d = vectors_2d[len(word_vectors):, ]
    df['vectors_2d'] = word_vectors_2d.tolist()

    plt.figure(figsize=(16, 16))
    cmap = get_cmap(num_cluster)

    for i in range(0, num_cluster):
        temp = df.loc[df['cluster-labels'] == i]
        points = temp['vectors_2d'].values
        x = list(zip(*points))[0]
        y = list(zip(*points))[1]
        labels = temp.index
        # rgb = np.random.rand(3, )
        plt.scatter(centroids_2d[i][0], centroids_2d[i][1], c=[cmap(i)], marker='X', s=10 ** 2)
        plt.scatter(x, y, c=[cmap(i)], alpha=0.9, label='Cluster' + str(i))
        # plt.pause(0.5)
        plt.legend(loc=4)

        for index, word in enumerate(labels):
            plt.annotate(word,
                         alpha=0.9,
                         xy=(x[index], y[index]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom',
                         size=8)
            # plt.pause(2.0)
    filename = os.path.join(filepath, 'kmeans_cluster_plots', 'cluster_plot')
    plt.savefig(filename)


def plot_kmeans_clusters_plotly(df, num_cluster, word_vectors, cluster_centroids, filepath):
    """Plot word clusters built based on KMeans Clustering of Word2Vec
    This method reduced vectors dimensions to 2-D using TSNE. These 2-D embeddings are then
    converted into a plotly graph.
    Read about TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Note: Clustering is only used to cross validate Word2Vec word-embeddings. It's not being used in dashboard

    Parameters
    ----------
    df: DataFrame
        Iterator of word features

    num_cluster: int
        Number of KMeans Clusters

    word_vectors: list
        Iterator of word embeddings

    cluster_centroids: list
        co-ordinates of KMeans Cluster centroids

    filepath: str
        Filename to save the plot
    """
    api_logger.info("Saving KMeans cluster plot to {0}".format(filepath))

    word_vectors = np.array(word_vectors)
    cluster_centroids = np.array(cluster_centroids)
    temp_arr = np.vstack((word_vectors, cluster_centroids))  # combining two for dimensionality reduction

    tsne_model = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3000, random_state=32)
    vectors_2d = tsne_model.fit_transform(temp_arr)

    vectors_2d = np.array(vectors_2d)
    word_vectors_2d = vectors_2d[:len(word_vectors), ]
    centroids_2d = vectors_2d[len(word_vectors):, ]
    df['vectors_2d'] = word_vectors_2d.tolist()

    fig = go.Figure(layout=dict(title='Plot of KMeans Clusters of words'))
    new_cmap = rand_cmap(num_cluster, type='bright', first_color_black=True, last_color_black=False, verbose=False)

    for i in range(0, num_cluster):
        temp = df.loc[df['cluster-labels'] == i]
        points = temp['vectors_2d'].values
        x = list(zip(*points))[0]
        y = list(zip(*points))[1]
        labels = temp.index

        api_logger.debug("Writing KMeans results to {0}".format(result_dir))
        save_kmeans_word_clusters(i, labels)
        fig.add_trace(go.Scatter(x=[centroids_2d[i][0]], y=[centroids_2d[i][1]],
                                 name='Centroid ' + str(i),
                                 mode='markers',
                                 marker=dict(size=20, symbol='x-dot', color='rgba' + str(new_cmap(i))),
                                 ))

        fig.add_trace(go.Scatter(x=x, y=y,
                                 name='Cluster ' + str(i),
                                 mode='markers',
                                 hovertemplate="<br>".join([
                                     "<b>(x, y):</b> %{x}, %{y}",
                                     f"<b>Cluster:</b> {str(i)}",
                                     "<b>Label:</b> %{text}"
                                 ]),
                                 marker=dict(size=10, color='rgba' + str(new_cmap(i))),
                                 text=labels[:], textposition='top right'))

    filename = os.path.join(filepath, 'kmeans_cluster_plots', 'kmeans_cluster_plot.html')
    plotly.offline.plot(fig, filename=filename)


def save_kmeans_word_clusters(cluster, labels):
    """Save 'Cluster Number: Words in the cluster' to a text file

    Parameters
    ----------
    cluster: int
        Cluster Number

    labels: list
        Iterator of word features
    """
    filename = os.path.join(result_dir, 'kmeans_word_clusters.txt')

    with open(filename, 'a') as file:
        file.write("KMeans Cluster output for HSD word embeddings\n")
        file.write('Cluster ' + str(cluster) + ': ' + ', '.join(labels) + '\n')

        file.write('\n')


def get_cmap(n, name='gist_rainbow'):
    """Take length of list and Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    Parameters
    ----------
    name: standard mpl colormap name. reference: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

    n: length of the distinct colour list.

    Returns
    ----------
    a function that maps each index from n to a distinct RGB color.
    """
    return plt.cm.get_cmap(name, n)


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks.

    Parameters
    ----------
    nlabels: int
        Number of labels (size of colormap)

    type: str, default='bright'
        'bright' for strong colors, 'soft' for pastel colors

    first_color_black: bool, default=True
        Option to use first color as black, True or False

    last_color_black: bool, default=False
        Option to use last color as black, True or False

    verbose: bool, default=False
        Prints the number of labels and shows the colormap. True or False

    Returns
    ----------
    colormap for matplotlib and plotly
    """

    from matplotlib.colors import LinearSegmentedColormap
    import colorsys

    if type not in ('bright', 'soft'):
        api_logger.warning("Please choose 'bright' or 'soft' for type")
        return

    if verbose:
        api_logger.info("Number of labels: {0}".format(str(nlabels)))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap
