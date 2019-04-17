import re
import math
import operator
import numpy as np
from ast import literal_eval
from collections import Counter
from utils import preprocess_transcripts

from nltk.stem import PorterStemmer


def get_concepts(transcript, tags=None, summary_length=3):
    """
    Get the concepts of the document based on the TED Talk tags or the high-
    frequency words of the transcript.

    :param transcript: List of transcript sentences.
    :param tags: List of transcript tags.
    :param summary_length: Lenth of the desired summary.
    :return concepts: List of prominent transcript concepts.
    :rtype: list(str)
    """
    # If tags are provided, extract concepts based on these, otherwise
    # extract concepts as the words having highest frequency in document.
    if tags is not None:
        concepts_struct = {}
        for w in tags:
            concepts_struct[w] = sum(1 for _ in re.finditer(r'\b%s\b' \
                           % re.escape(w), ' '.join(transcript)))
        sorted_struct = {k:v for (k,v) in sorted(concepts_struct.items(),\
                         key=operator.itemgetter(1), reverse=True) if v > 0}
        concepts = [k for k,_ in sorted_struct.items()]
    else:
        word_frequencies = Counter(' '.join(transcript).split()).most_common()
        concepts = [word_frequencies[j][0] for j in range(summary_length)]
    return concepts


def topic_representation(transcript, concepts):
    """
    Computes the topic representation matrix for the document.

    :param transcript: List containing transcript sentences.
    :param concepts : List containing the main topics of transcript.
    :return X: Topic representation matrix.
    :rtype: array_type
    """
    if len(concepts) == 0:
        return np.zeros((1,len(transcript)))

    total_docs = len(transcript)
    X = np.zeros((len(concepts), total_docs))

    for j in range(X.shape[0]):
        sentences_with_concept = 0
        for k in range(X.shape[1]):
            sentence = transcript[k].split()
            word_count = sum(1 for _ in re.finditer(r'\b%s\b' \
                           % re.escape(concepts[j]), transcript[k]))
            if word_count > 0:
                sentences_with_concept += 1
            # First store the tf(concept).
            X[j,k] = word_count/len(sentence)
        # Compute the idf
        if sentences_with_concept == 0:
            print(concepts, j)
        idf = math.log2(total_docs/sentences_with_concept)
        X[j,:] * idf

    return X


def gong_liu(Vh, transcript):
    """
    Implements Gong and Liu's sentence selection method.

    :param Vh: Right singular vectors of the topic representation matrix.
    :param transcript: List of transcript sentences.
    :return summary_indices: Row indices of the selected sentences (column
        indices of Vh).
    :rtype: list(int)
    """
    summary_indices = []
    for j in range(Vh.shape[0]):
        I, _ = max(enumerate(Vh[j]), key=operator.itemgetter(1))
        summary_indices.append(I)
    return summary_indices


def cross(Vh, transcript):
    """
    Implements the cross method for sentence selection.

    :param Vh: Right singular vectors of the topic representation matrix.
    :param transcript: List of transcript sentences.
    :return summary_indices: Row indices of the selected sentences (column
        indices of Vh).
    :rtype: list(int)
    """
    # Zero sentence scores less than the concepts average.
    for j in range(Vh.shape[0]):
        Vh[j,:] = np.absolute(Vh[j,:])
        avg_score = np.average(Vh[j,:])
        for jj in range(len(Vh[j,:])):
            if Vh[j,jj] <= avg_score:
                Vh[j,jj] = 0

    # Compute sentence lengths.
    sentence_lengths = [np.sum(Vh[:,j]) for j in range(Vh.shape[0])]
    summary_indices = []
    for _ in range(Vh.shape[0]):
        I, _ = max(enumerate(sentence_lengths), key=operator.itemgetter(1))
        summary_indices.append(I)
        sentence_lengths[I] = 0
    return summary_indices


def summarize_text(df_row, summary_length, method=cross, stem_words=False,
                   use_tag_concepts=True):
    """
    Summarizes input document.

    :param df_row: Single-record dataframe containing the document and tags.
    :param summary_length: Desired length of the returned summary.
    :param method: Sentence selection function.
    :param stem_words: Bool to signal stemming words or not.
    :param use_tag_concepts: Bool to signal whether to use tags as concepts.
    :return summary: Text summary.
    :rtype: str
    :return concepts: List of derived concepts.
    :rtype: list(str)
    """
    full_transcript = preprocess_transcripts(df_row, False)[0]
    clean_transcript = preprocess_transcripts(df_row)[0]

    tags = None
    if use_tag_concepts:
        tags = literal_eval(df_row['tags'].iloc[0])

    concepts = get_concepts(clean_transcript, tags, summary_length)
    if stem_words:
        ps = PorterStemmer()
        concepts = [ps.stem(w) for w in concepts]

    X = topic_representation(clean_transcript, concepts)
    _, s, vh = np.linalg.svd(X)

    summary_indices = method(vh[:len(concepts),:], clean_transcript)
    summary = ''
    for idx in summary_indices:
        summary += full_transcript[idx] + '. '
    return summary, concepts
