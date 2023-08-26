
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from utilities import *
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import hdbscan
import umap
import json
import sys
import ast
import os


sys.setrecursionlimit(1000000)
max_clusters = {} # max_clusters by min_sample


class ClusterEmbeddings:
    """
    Cluster the text embeddings of ad's textual content with HDBSCAN
    embedding modality: text only # 24930 cases
    
    Parameters
    ----------
    base_dir = base directory of output
    embeddings_dir: BERT-based text embeddings, obtained after running bert-embeddings.py
    reduce_dimension_flag: boolean to evaluate if dimensionality reduction to perform
    eps: DBSCAN parameter: eps distance
    min_samples: DBSCAN parameter: mininum samples to define a core sample

    Functions
    ----------
    PCA_reduce: perform PCA dimentionality reduction
    UMAP_reduce: perform UMAP dimentionality reduction
    cluster_HDBSCAN: perform HDBSCAN clustering on text embeddings

    Outputs
    ----------
    hdbscan_text.npy: predicted cluter labels (same order as the ad's text embeddings), saved in .npy 
    """
    def __init__(self, base_dir, embeddings_path, reduce_dimension_flag: bool=False):
        self.BASE = base_dir
        self.dtype = np.float32
        self.data = np.load(embeddings_path)
        self.umap_flag = reduce_dimension_flag

    def PCA_reduce(self, embeddings, new_dim=128):
        reduced_embeddings = PCA(n_components=new_dim).fit_transform(embeddings)
        return reduced_embeddings

    def UMAP_reduce(self, embeddings, new_dim=2):
        # reduced_embeddings = umap.UMAP(random_state=2023).fit_transform(embeddings) # original
        # UMAP(n_neighbors=15, n_components=128, min_dist=0.0, metric='euclidean') # detailed
        reduced_embeddings = umap.UMAP(n_components=128, n_neighbors=15, min_dist=0.0, metric='euclidean', random_state=2023).fit_transform(embeddings) # to try
        return reduced_embeddings

    def cluster_HDBSCAN(self, modality, min_cluster_size=5):
        embeddings = np.asarray(self.data["text"], dtype=self.dtype)
        if self.umap_flag:
            print(embeddings.shape)
            X = self.UMAP_reduce(embeddings)
            # X = self.PCA_reduce(embeddings)
            print(X.shape)
        else:
            X = embeddings

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True).fit(X)
        '''
        # Visualizing Clusters using in-built method: Not effective due to large # of data points
        fig = plt.figure(figsize=(75, 50))
        clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.75, node_size=500, edge_linewidth=20)
        cluster_labels = clusterer.labels_
        plt.savefig(os.path.join(self.BASE, 'textual_clusters_mst_16.png'), dpi = 300)
        print("Figure saved!")
        '''

        noise_indices = np.where(cluster_labels==-1)[0]
        noise_mask = np.isin(np.arange(X.shape[0]), noise_indices)
        print(f"There are {len(noise_indices)} noise data.")
        print(f"There are {len(np.unique(cluster_labels))} clusters.")
        
        global max_clusters;
        if str(min_cluster_size) not in max_clusters.keys():
            max_clusters[str(min_cluster_size)] = 0
        if len(np.unique(cluster_labels)) >= int(max_clusters[str(min_cluster_size)]):
            max_clusters[str(min_cluster_size)] = len(np.unique(cluster_labels))
            print(min_cluster_size, len(noise_indices), len(np.unique(cluster_labels)))
        
        np.save(f"{self.BASE}/hdbscan-{modality}", cluster_labels)



class ClusterAnalysis:
    """
    Analyse clusters resulting from the HDBSCAN clustering

    Parameters
    ----------
    base_dir: output directory to save results
    embeddings_path: ad's text embeddings filepath
    cluster_labels_path: path to cluster labels file, obtained from class ClusterEmbeddings
    datafile_path: File containing ad texts for all ads in our dataset

    Functions
    ----------
    extract_keywords: extract_keywords using three different extraction methods
    automatic_annotation: annotate each cluster with key-phrases
    annotation_visualization: tsne visualization of the top-100 clusters
    build_cluster_graph: build a graph with cluster centroids as node
    read_by_cluster_id: get indices where cluster label matches the current cluster id
    retrieve_text_by_clusterID: Read ad texts for input indices from read_by_cluster_id
    retrieve_text_by_similarity: Compute centroid embedding of a cluster by taking mean of text embeddings inside it and find topk most similar ad_texts closest to centroid embedding
    process_keyphrase: Preprocess keyphrases
    embedding_annotation_mapping: Annotating each text embedding with keyphrases

    Outputs
    ----------
    cluster_annotation.xlsx
    cluster_details.xlsx
    tsne_text.pdf
    graph.gexf
    """

    def __init__(self, base_dir, embeddings_path, cluster_labels_path, datafile_path):
        self.BASE = base_dir
        self.save_dir = base_dir
        embeddings = np.load(embeddings_path, allow_pickle=True)
        self.text_embeddings = embeddings["text"]
        self.cluster_labels =  np.load(cluster_labels_path, allow_pickle=True)
        self.counter = Counter(self.cluster_labels).most_common() # [(real_cluster_id, num), ...]
        assert len(self.cluster_labels) == len(self.text_embeddings)

        with open(datafile_path, "r") as file:
            self.ad_texts = file.readlines()
            file.close()
    
    def read_by_cluster_id(self, cluster_id):
        real_id = self.counter[cluster_id][0]
        indices = np.where(self.cluster_labels==real_id)[0]
        return indices # indices of (text) in current cluster

    def retrieve_text_by_clusterID(self, cluster_id, if_topk=False):
        indices = self.read_by_cluster_id(cluster_id)
        if len(indices) >=300 and if_topk:
            indices = np.random.choice(indices, 300, replace=False)
        text = [json.loads(line)["ad_text_translated"] for line in np.array(self.ad_texts)[indices]]
        return indices, text
    
    def retrieve_text_by_similarity(self, cluster_id, topk=300):
        indices = self.read_by_cluster_id(cluster_id)
        centroid = np.mean(self.text_embeddings[indices], axis=0)
        centroid = np.expand_dims(centroid, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        norm_text_embeddings = self.text_embeddings / np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        similarities = centroid @ norm_text_embeddings.T # [1, len(embeddings)]
        topk_indices = np.argsort(np.squeeze(similarities))[::-1][:topk] # indices of topk similarites
        text = [json.loads(line)["ad_text_translated"] for line in np.array(self.ad_texts)[topk_indices]]
        return topk_indices, text

    # def automatic_annotation(self, mode="by_similarity"):
    def automatic_annotation(self, mode="by_index"):
        select_clusters_ids = np.array([(i, j) for (i, j) in self.counter if j >= 1 and j <= 100000])
        res, full_ouput = [], []
        for real_id, num in tqdm(select_clusters_ids):
            cnt = 0
            indices = np.where(self.cluster_labels==real_id)[0]
            nominal_id = self.counter.index((real_id, num))
            if mode == "by_index":
                _, texts = self.retrieve_text_by_clusterID(cluster_id=nominal_id, if_topk=False)
            elif mode == "by_similarity":
                _, texts = self.retrieve_text_by_similarity(cluster_id=nominal_id, topk=300)
            else:
                raise Exception("wrong mode!")
            text = [" ".join(clean_text([ad])) for ad in texts]

            keyphrases = extract_keywords_keybert(text=" ".join(text), extract_rule="ngram", top_n=3)
            res.append([nominal_id, keyphrases, len(indices)])
            for txt in texts:
                cnt += 1
                full_ouput.append([nominal_id, cnt, txt, keyphrases, len(indices)])
        
        save_df = pd.DataFrame(res, columns=["nominal_id",  "ngram",  "num_samples", "hate_score"])
        save_df.to_excel(f"{self.save_dir}/cluster-annotation.xlsx")
        save_full_df = pd.DataFrame(full_ouput, columns=["nominal_id",  "cluster_sub_index", "ad_content", "ngram", "cluster_size"])
        save_full_df.to_excel(f"{self.save_dir}/cluster-details.xlsx", index=False)
        self.annotation_file = f"{self.save_dir}/cluster-annotation.xlsx"
    
    def process_keyphrase(self, lst):
        noise_lst = ["t.co/"]
        phrase = None
        new_lst = []
        for _phrase in lst:
            if all(noise not in _phrase for noise in noise_lst):
                new_lst.append(_phrase)
        try: 
            for _phrase in new_lst:
                if len(_phrase.split())==3:
                    phrase = _phrase
                    break
            else:
                phrase = new_lst[0]
        except:
            phrase = lst[1]
        tokens = phrase.split()
        return "-".join(clean_text(tokens))

    def embedding_annotation_mapping(self, modality="text", keyphrase_method="ngram"):
        annotations_df = pd.read_excel(self.annotation_file)
        cluser_ids = annotations_df["nominal_id"].values
        unique_embeddings, cluster_keyphrases, size = [], [], [], []
        for cluster_id in cluser_ids:
            indices = self.read_by_cluster_id(cluster_id)
            if modality == "text":
                cluster_embeddings = self.text_embeddings[indices]
            elif modality == "fused":
                cluster_embeddings = self.image_embeddings[indices] + self.text_embeddings[indices]
            unique_embeddings.append(np.mean(cluster_embeddings, axis=0))
            row_in_df = annotations_df.loc[annotations_df["nominal_id"]==cluster_id]
            _annotation = row_in_df[keyphrase_method].values[0]
            _annotation = ast.literal_eval(_annotation)
            keyphrase = self.process_keyphrase(_annotation)
            cluster_keyphrases.append(keyphrase)
            size.append(int(row_in_df["num_samples"].values[0]))
        unique_embeddings = np.array(unique_embeddings)
        cluster_keyphrases = np.array(cluster_keyphrases)
        size = np.array(size)
        return unique_embeddings, cluster_keyphrases

    def annotation_visualization(self, modality="text", keyphrase_method="ngram", perplexity=10, topk=100):
        unique_embeddings, cluster_keyphrases = self.embedding_annotation_mapping(modality=modality, keyphrase_method=keyphrase_method)
        # unique_embeddings, cluster_keyphrases = unique_embeddings[:topk], cluster_keyphrases[:topk]
        unique_embeddings, cluster_keyphrases = unique_embeddings[:], cluster_keyphrases[:]

        n_components = 2
        tsne = TSNE(n_components=n_components, 
                    perplexity=perplexity,
                    n_iter=10000,
                    metric="cosine",
                    random_state=2023)
        pos = tsne.fit_transform(np.array(unique_embeddings))
        labels = cluster_keyphrases

        fig = plt.figure(figsize=(20, 20), frameon=False)
        K, FONTSIZE = 15, 20
        scatter_size = np.arange(1, K*len(labels), K)[::-1]

        for i, la in enumerate(labels):
            plt.scatter(pos[i,0], pos[i,1], s=scatter_size[i], alpha=0.4, edgecolor="white")
            # For plotting bounding box for related cluster in TSNE output
            if la in ["macbook-apple", "samsung-fold"]:
                plt.annotate(la, (pos[i,0], pos[i,1]), c="red", fontsize=FONTSIZE, bbox=dict(facecolor="none", edgecolor="red"))
            else:
                plt.annotate(la, (pos[i,0], pos[i,1]),fontsize=FONTSIZE)
        plt.axis("off")
        # plt.subplots_adjust(right=1.2)
        # plt.show()
        fig.savefig(f"{self.save_dir}/tsne-{modality}.pdf", bbox_inches="tight", dpi=2000)

        return pos, cluster_keyphrases
        
    def build_cluster_graph(self, modality="text", keyphrase_method="ngram", save_graph=True):
        unique_embeddings, cluster_keyphrases = self.embedding_annotation_mapping(modality=modality, keyphrase_method=keyphrase_method)
        unique_embeddings /= np.linalg.norm(unique_embeddings, axis=1, keepdims=True)
        cosine_matrix = unique_embeddings @ unique_embeddings.T
        threds = []
        for row in cosine_matrix:
            threds.append(np.percentile(row, 98.5))
        threds = np.array(threds)
  
        G = nx.Graph()
        for idx in range(cosine_matrix.shape[0]):
            G.add_node(idx, label=cluster_keyphrases[idx])
        
        for idx in range(cosine_matrix.shape[0]):
            indices = np.where(cosine_matrix[idx]>=threds[idx])[0]
            for ndx in indices:
                if (idx, ndx) not in G.edges():
                    G.add_edge(idx, ndx, weight=cosine_matrix[idx][ndx])
        if save_graph:
            nx.write_gexf(G, f"{self.save_dir}/graph.gexf")
        return G
    


def perform_clustering():

    bert_base_dir = "/INET/socialnets3/static00/yvekaria/bert_data"
    bert_embeddings_path = os.path.join(bert_base_dir, "bert-embeddings.npz")

    ce = ClusterEmbeddings(bert_base_dir, bert_embeddings_path, "True")
    
    '''
    for min_cluster_size in range(100, 4, -1):
            ce.cluster_HDBSCAN("text", min_cluster_size)
    
    global max_clusters;
    print(max_clusters)
    
    # min_cluster_size: clusters
    # {'5': 913, '6': 697, '7': 556, '8': 412, '9': 365, '10': 318, '100': 8, '99': 9, '98': 9, '97': 10, '96': 9, '95': 10, '94': 9, '93': 9, '92': 10, '91': 8, '90': 11, '89': 10, '88': 10, '87': 11, '86': 10, '85': 9, '84': 10, '83': 9, '82': 10, '81': 11, '80': 9, '79': 8, '78': 11, '77': 11, '76': 10, '75': 13, '74': 11, '73': 13,
       '72': 12, '71': 12, '70': 11, '69': 12, '68': 14, '67': 13, '66': 16, '65': 15, '64': 17, '63': 10, '62': 13, '61': 16, '60': 14, '59': 15, '58': 16, '57': 10, '56': 15, '55': 17, '54': 15, '53': 16, '52': 20, '51': 17, '50': 17, '49': 24, '48': 23, '47': 19, '46': 22, '45': 20, '44': 21, '43': 21, '42': 21, '41': 25, '40': 21, '39': 26, '38': 23, '37': 28, 
       '36': 22, '35': 32, '34': 36, '33': 28, '32': 35, '31': 27, '30': 33, '29': 42, '28': 72, '27': 34, '26': 43, '25': 39, '24': 40, '23': 43, '22': 44, '21': 110, '20': 61, '19': 63, '18': 58, '17': 62, '16': 170, '15': 70, '14': 175, '13': 213, '12': 88, '11': 270}
    '''

    ce.cluster_HDBSCAN("text", min_cluster_size=16) # (24929, 512) --> (24929, 2) --> 16 (min cluster size) 10039 (noise) 150 (clusters)

    return



def analyze_clusters():

    bert_base_dir = "/INET/socialnets3/static00/yvekaria/bert_data"
    bert_embeddings_path = os.path.join(bert_base_dir, "bert-embeddings.npz")
    cluster_labels_path = os.path.join(bert_base_dir, "hdbscan-text.npy")
    datafile_path = os.path.join(bert_base_dir, "ad-tweet-data.txt")
    
    ca = ClusterAnalysis(bert_base_dir, bert_embeddings_path, cluster_labels_path, datafile_path)

    # Annotating each cluster with keyphrases extracted from the text
    ca.automatic_annotation()

    # tsne visualization of the top-100 clusters
    # Each cluster is represented with its centroid (text/fused) embedding
    ca.annotation_visualization(modality="text", perplexity=4)

    # Building a graph with cluster centroids as node
    # The output .gexf can be processed with gephi
    ca.build_cluster_graph()

    return



def main():

    perform_clustering()

    analyze_clusters()

    return



if __name__ == "__main__":

    main()
