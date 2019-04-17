import io
import re
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from PyRouge.pyrouge import Rouge
from pythonrouge.pythonrouge import Pythonrouge

import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

# One-time calls
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize


def read_data(record_count=None):
    """
    Reads TED Talk data from `ted_main.csc` and `transcripts.csv`.

    :param record_count: Sets number of records to be return in dataframe.
    :return: Dataframe with columns, `description`, `tags`, `title`,
        `transcript`, `t_cnt`, `d_cnt`.
    :rtype: Pandas dataframe
    """
    main = pd.read_csv('./Data/ted_main.csv')
    transcripts = pd.read_csv('./Data/transcripts.csv')
    df = pd.merge(main, transcripts, on='url')
    df.drop(df[df.transcript.str.contains('(Music)')].index, inplace=True)
    df['t_cnt'] = df.transcript.apply(lambda x: len(sent_tokenize(x)))
    df['d_cnt'] = df.description.apply(lambda x:len(sent_tokenize(x)))
    df = df[df['d_cnt'] > 1]
    return df[['description','tags','title','transcript','t_cnt','d_cnt']]\
            .dropna().reset_index(drop=True)[:record_count]


def preprocess_transcripts(df,
                           clean_transcript_data=True,
                           remove_stopwords=True,
                           stem_words=False,
                           df_field='transcript'):
    """
    Processes raw transcript data.

    :param df: Pandas dataframe as returned from `read_data()`.
    :param clean_transcript_data: Bool signaling to return cleaned or uncleaned
        sentences.
    :param remove_stopwords: Bool signaling to remove stopwords or not.
    :param stem_words: Bool signaling to stem words or not.
    :param df_field: Indicates which dataframe text field to process.
    :return processed_transcripts: List of transcripts split into lists of
        sentences.
    :rtype: list(str)
    """
    transcripts = []
    for s in df[df_field]:
        transcripts.append(sent_tokenize(s))

    if not clean_transcript_data:
        processed_transcripts = []
        for t in transcripts:
            pt = pd.Series(t).str.replace(r"(\().*(\))|([^a-zA-Z'])",' ')
            pt = [' '.join([j.lower() for j in w.split()]) for w in pt]
            processed_transcripts.append(list(filter(None, pt)))
        return processed_transcripts

    else:
        # Remove numbers, punctuation and stop words.
        processed_transcripts = []
        stop_words = stopwords.words('english')
        if not remove_stopwords:
            stop_words = []
        ps = PorterStemmer()
        for t in transcripts:
            pt = pd.Series(t).str.replace(r"(\().*(\))|([^a-zA-Z])",' ')
            if stem_words:
                pt = [' '.join([ps.stem(j.lower()) for j in w.split()\
                                if j not in stop_words]) for w in pt]
            else:
                pt = [' '.join([j.lower() for j in w.split()\
                                if j not in stop_words]) for w in pt]
            processed_transcripts.append(list(filter(None, pt)))
        return processed_transcripts


def load_wordvectors(load_vec_file=False):
    """
    Loads and returns fastText word embedding.

    :param load_vec_file: Bool indicating whether to load `.txt` or `.bin` file.
    :return word_embeddings: fastText word embedding.
    :rtype: map(str): array_type
    """
    if load_vec_file:
        """From https://fasttext.cc/docs/en/english-vectors.html"""
        fname = './Data/wiki-news-300d-1M.vec'
        fin = io.open(fname,'r',encoding='utf-8',newline='\n',errors='ignore')
        n, d = map(int, fin.readline().split())
        word_embeddings = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            word_embeddings[tokens[0]] = map(float, tokens[1:])
    else:
        from gensim.models import FastText
        word_embeddings = FastText.load_fasttext_format('./Data/cc.en.300.bin')
    return word_embeddings


def embed_sentences(transcript, wordvectors=None):
    """
    Embed transcript sentences.

    :param transcript: List of transcripts split into lists of sentences.
    :param wordvectors: Word embedding (as returned from `load_wordvectors()`).
    :return sentence_vectors: List of sentence vectors of the input transcript.
    :rtype: list(array_type)
    :return sentence_indices: Indices of the sentences return from the list,
        `transcript`.
    :rtype: list(int)
    """
    if wordvectors is None:
        wordvectors = load_wordvectors()
    sentence_vectors = []
    sentence_indices = []
    for j, s in enumerate(transcript):
        vl = []
        if len(s.split()) != 0:
            try:
                vl = [wordvectors.wv[w] for w in s.split()]
                v = np.asarray(np.sum(vl, axis=0))/(len(vl)+0.001)
            except:
                v = np.random.rand(300)
        else:
            v = np.random.rand(300)
        if len(v) >= 300:
            sentence_vectors.append(v[:300])
            sentence_indices.append(j)
    return sentence_vectors, sentence_indices


class Evaluation:
    """
    Evaluation class with evaluation metric tests, `rouge_l`, `rouge_n`, and
    `cos_similarity`.
    """
    def __init__(self):
        pass

    @staticmethod
    def rouge_l(S, I):
        r = Rouge()
        [precision, recall, f_score] = r.rouge_l([S], [I])
        return f_score

    @staticmethod
    def rouge_n(S, I, N, beta=1.25):
        sl = S.split()
        sngrams = []
        for j in range(len(sl)-N+1):
            ngram = []
            for jj in range(N):
                ngram.append(sl[jj+j])
            sngrams.append(ngram)

        recovered_ngrams = 0
        for ngram in sngrams:
            ng = ' '.join(ngram)
            if ng in I:
                recovered_ngrams += 1

        P = recovered_ngrams/len(sngrams)
        R = recovered_ngrams/(len(I.split())-N+1)
        F_score = (1 + beta**2) * R * P / (beta**2 * P + R + 0.0001)
        return F_score

    @staticmethod
    def cos_similarity(S, I, embedding):
        summary_vecs = embed_sentences([w for w in S.split('. ')],embedding)[0]
        ideal_vecs = embed_sentences([w for w in I.split('. ')],embedding)[0]
        v1 = np.average(summary_vecs, axis=0)
        v2 = np.average(ideal_vecs, axis=0)
        return cosine_similarity(v1.reshape((1,300)),v2.reshape((1,300)))[0,0]


def get_tag_percentages(df):
    """
    Computes the percentage of tags used per transcript and the percentage of
    tag occurences per transcript and writes results to `processed_data.csv`.

    :param df: Pandas dataframe as returned from `read_data()`.
    """
    tag_use_percentages = []
    tag_occurence_percent = []

    for j in range(len(df)):
        transcript = ' '.join(preprocess_transcripts(df.iloc[[j]])[0])
        tags = literal_eval(df['tags'].iloc[j])

        tag_occurrence = 0
        tag_frequency = 0
        for t in tags:
            if re.search(r'\b' + t + r'\b', transcript):
                tag_occurrence += 1
                tag_frequency += sum(1 for _ in re.finditer(r'\b%s\b' \
                           % re.escape(t), transcript))

        tag_use_percentages.append(tag_occurrence/len(tags))
        tag_occurence_percent.append(tag_frequency/len(transcript.split()))

    df['percent_tag_used'] = tag_use_percentages
    df['percent_tag_occurred'] = tag_occurence_percent
    df.to_csv('processed_data.csv')


def plot_evaluations(df):
    """
    Plots the evaluation results and transcript length (normalized) in
    increasing order for each transcript.

    :param df: Pandas dataframe as returned from `read_data()`.
    """
    import matplotlib.pyplot as plt

    df = df.sort_values('t_cnt_nrm')
    df = df.reset_index()
    df = df.drop(columns='index')

    ax1 = plt.gca()
    df.plot(kind='line', y='t_cnt_nrm', use_index=True, ax=ax1)
    df.plot(kind='line', y='gtc7_cs', use_index=True, ax=ax1)
    df.plot(kind='line', y='gnt7_cs', use_index=True, ax=ax1)
    plt.show()

    ax2 = plt.gca()
    df.plot(kind='line', y='t_cnt_nrm', use_index=True, ax=ax2)
    df.plot(kind='line', y='ctc7_cs', use_index=True, ax=ax2)
    df.plot(kind='line', y='cnt7_cs', use_index=True, ax=ax2)
    plt.show()
