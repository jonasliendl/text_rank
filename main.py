import os
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

TOP_K = 5
RESULTS_DIR = "results"

class SentenceSplitter:
    def __init__(self, text):
        self.text = text

    def split(self):
        sentences = [s.strip() for s in self.text.replace("\n", " ").split('.') if s]
        return [s + '.' for s in sentences]

class VectorizerTFIDF:
    def __init__(self, sentences):
        self.sentences = sentences
        self.vectorizer = TfidfVectorizer()

    def transform(self):
        tfidf_matrix = self.vectorizer.fit_transform(self.sentences).toarray()
        np.save(os.path.join(RESULTS_DIR, 'tfidf_vectors.npy'), tfidf_matrix)
        return tfidf_matrix

class VectorizerEmbedding:
    def __init__(self, sentences, model_name='all-MiniLM-L6-v2'):
        self.sentences = sentences
        self.model = SentenceTransformer(model_name)

    def transform(self):
        embeddings = self.model.encode(self.sentences)
        np.save(os.path.join(RESULTS_DIR, 'embeddings.npy'), embeddings)
        return embeddings

class GraphBuilder:
    def __init__(self, vectors):
        self.vectors = vectors

    def build(self):
        sim_matrix = np.dot(self.vectors, self.vectors.T)
        np.save(os.path.join(RESULTS_DIR, 'sim_matrix.npy'), sim_matrix)
        graph = nx.from_numpy_array(sim_matrix)
        return graph

    def plot(self, graph, title, filename):
        plt.figure()
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=500)
        plt.title(title)
        plt.savefig(os.path.join(RESULTS_DIR, filename))
        plt.close()

class TextRankSummarizer:
    def __init__(self, graph, sentences):
        self.graph = graph
        self.sentences = sentences

    def summarize(self):
        scores = nx.pagerank(self.graph)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_sentences = [self.sentences[idx] for idx, _ in ranked[:TOP_K]]
        return top_sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    args = parser.parse_args()

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    splitter = SentenceSplitter(text)
    sentences = splitter.split()

    tfidf_vec = VectorizerTFIDF(sentences)
    tfidf_matrix = tfidf_vec.transform()

    embed_vec = VectorizerEmbedding(sentences)
    embed_matrix = embed_vec.transform()

    tfidf_graph = GraphBuilder(tfidf_matrix).build()
    GraphBuilder(tfidf_matrix).plot(tfidf_graph, 'TF-IDF Similarity Graph', 'tfidf_graph.png')

    embed_graph = GraphBuilder(embed_matrix).build()
    GraphBuilder(embed_matrix).plot(embed_graph, 'Embedding Similarity Graph', 'embed_graph.png')

    tfidf_summary = TextRankSummarizer(tfidf_graph, sentences).summarize()
    embed_summary = TextRankSummarizer(embed_graph, sentences).summarize()

    print('TF-IDF Top Sentences:')
    for s in tfidf_summary:
        print(s)

    print('\nEmbedding Top Sentences:')
    for s in embed_summary:
        print(s)

if __name__ == '__main__':
    main()
