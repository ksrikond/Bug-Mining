import os
from sklearn import cluster
from sklearn.metrics import silhouette_score

from src.utils.logger_config import api_logger

root_path = os.getcwd()
src_path = os.path.join(root_path, 'src')
datafiles_path = os.path.join(src_path, 'data', 'data_files')
visualization_path = os.path.join(src_path, 'visualization')


class Cluster:
    """KMeans Clustering for HSD data. HSD Mining Analysis uses Word2Vec as the vectorization process
    to convert text data into a word embedding matrix.
    Note: KMeans Clustering is used to verify the process of generating word embeddings.

    Parameters
    ----------
    n_clusters: int, default=3
        Number of clusters initially selected
    """
    def __init__(self, n_clusters=3):
        self.num_clusters = n_clusters

    def cluster_data(self, word_vectors):
        """Cluster word embeddings

        Parameters
        ----------
        word_vectors: Series as ndarray or ndarray-like depending on the dtype
            Word Embeddings

        Returns
        ----------
        Cluster labels and centroids
        """
        api_logger.info("******Clustering using KMeans******")
        api_logger.info("Numbers of Clusters selected initially: {0}".format(self.num_clusters))
        # change verbose to 0 to stop printing the information
        max_silhouette, labels, centroids = -1, None, None

        for i in range(2, self.num_clusters+1):
            kmeans = cluster.KMeans(n_clusters=i, max_iter=300, verbose=0, n_init=50, random_state=10)
            kmeans.fit(word_vectors)

            labels_tmp, centroids_tmp = kmeans.labels_, kmeans.cluster_centers_

            silhouette_avg = silhouette_score(word_vectors, labels_tmp)
            api_logger.info("For n_clusters = {0} The average silhouette_score is : {1}".format(i, silhouette_avg))
            if silhouette_avg > max_silhouette:
                max_silhouette = silhouette_avg
                labels, centroids = labels_tmp, centroids_tmp
                self.num_clusters = i  # selecting number of clusters with highest silhouette score

        api_logger.info("Finished clustering using KMeans, returning cluster labels and centroids details for "
                        "each cluster")
        api_logger.info("Number of clusters selected after silhouette score analysis: {0}".format(self.num_clusters))
        return labels, centroids


# if __name__ == '__main__':
#     filename = os.path.join(datafiles_path, 'word2_vec_data', 'word2vec_description.csv')
#     df = pd.read_csv(filename, index_col=0)
#     vocab, vectors = df.index, df.values
#     num_clusters = 3
#     # print(vocab, vectors)
#     cluster_labels, cluster_centroids = cluster_data(vectors, NUM_CLUSTERS=num_clusters)
#     df['cluster-labels'] = cluster_labels
#     # for i in range(0, num_clusters):
#     #     print(df['cluster-labels' == i])
#     _util.plot_kmeans_clusters_plotly(df, num_clusters, vectors, cluster_centroids, filepath=visualization_path)
