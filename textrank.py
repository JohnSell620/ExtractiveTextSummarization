import numpy as np
import networkx as nx
from utils import preprocess_transcripts, embed_sentences
from sklearn.metrics.pairwise import cosine_similarity


def get_sentence_similarities(sentence_vectors):
    """
    Computes the sentence similarity matrix of the TextRank algorithm,
    transforms the matrix to graph representation, and computes the Pagerank
    scores of each node.

    :param sentence_vectors: The sentence embeddings for a single transcript.
    :return Dictionary of nodes with PageRank as value.
    :rtype dict(array_type): float
    """
    l = len(sentence_vectors)
    similarity_matrix = np.zeros((l,l))
    for j in range(l):
        for k in range(l):
            if j != k:
                similarity_matrix[j][k] = cosine_similarity(\
                             sentence_vectors[j].reshape((1,300)),\
                             sentence_vectors[k].reshape((1,300)))[0,0]
    nx_graph = nx.from_numpy_array(similarity_matrix)
    return nx.pagerank(nx_graph)


def summarize_text(df_row, summary_length, embedding=None):
    """
    Summarizes transcript of data record.

    :param df_row: Single-record Pandas dataframe.
    :param summary_length: Desired length of the returned document summary.
    :param embedding: fastText word embedding.
    :return summary: Text summary of document.
    :rtype: str
    """
    transcript = preprocess_transcripts(df_row)[0]
    sentence_vectors, sentence_indices = embed_sentences(transcript, embedding)
    sentence_similarities = get_sentence_similarities(sentence_vectors)

    full_transcript = np.array(preprocess_transcripts(df_row, False)[0])
    new_transcript = list(full_transcript[sentence_indices])
    ranked_sentences = sorted(((sentence_similarities[j], s) for j, s in\
                               enumerate(new_transcript)), reverse=True)
    summary = ''
    for j in range(min(summary_length, len(ranked_sentences))):
        summary += ranked_sentences[j][1] + '. '
    return summary
