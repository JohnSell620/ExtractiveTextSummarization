import sys
import lsa
import textrank
import numpy as np
import pandas as pd
from ast import literal_eval
from utils import read_data, Evaluation, get_tag_percentages, \
                    preprocess_transcripts, load_wordvectors


def summarize(df_row, summary_length, use_tag_concepts=True, embedding):
    """
    Handle for summarization method to use (LSA or TextRank).

    :param df_row: Single-record dataframe containing transcript and tags.
    :param summary_length: Desired length of returned summary.
    :param embedding: fastText word embedding.
    :return method.summarize_text(): Text summarization method.
    :rtype: method
    """
    if embedding is None:
        return lsa.summarize_text(df_row, summary_length,
                                  use_tag_concepts=use_tag_concepts)
    elif method is textrank:
        return textrank.summarize_text(df_row, summary_length, embedding)
    else:
        print('In main.py: summarize(): Unrecognized method.')
        sys.exit(-1)


def drop_noconcept_recs(df):
    """
    Drop dataframe records whose transcripts do not mention any tags.

    :param df: Pandas dataframe containing TED Talk data.
    :return: Processed dataframe.
    :rtype: Pandas dataframe
    """
    C = []
    for j in range(len(df)):
        dfj = df.iloc[[j]]
        ctj = preprocess_transcripts(dfj)[0]
        tagsj = literal_eval(dfj['tags'].iloc[0])
        if tagsj is None:
            C.append([])
        else:
            conceptsj = lsa.get_concepts(ctj, tagsj)
            if len(conceptsj) > 2:
                C.append(conceptsj)
            else:
                C.append([])
    df['concepts'] = C
    df = df[df.astype(str)['concepts'] != '[]'].reset_index(drop=True)
    return df


def eval_avgs_against_descriptions(embedding=None):
    """
    Compute ROUGE-L and cosine similarity evaluation measures where ideal
    summaries are taken as the TED Talk descriptions.

    :param embedding: fastText word embedding.
    :return: List containing averages of precision, recall, F-score, and cosine
        similarity over 50 documents.
    :rtype: list(float)
    """
    # Get records where one or more tag is present in transcript.
    df = read_data(75)
    df = drop_noconcept_recs(df)[:50]

    results = []
    for j in range(len(df)):
        s = summarize(df.iloc[[j]], df['d_cnt'].iloc[[j]][j])
        ideal = preprocess_transcripts(df.iloc[[j]],
                                       clean_transcript_data=False,
                                       df_field='description')
        rl = Evaluation.rouge_l(s, ideal[0][0])
        cs = Evaluation.cos_similarity(s, ideal[0][0])
        results.append([rl, cs])

    # Average evaluation scores over number of dataframe records.
    results = np.asarray(results)
    rlresults = results[:,0]
    cossim_results = results[:,1]
    avg_prec = np.average([rlresults[j][0] for j in range(results.shape[0])])
    avg_recall = np.average([rlresults[j][1] for j in range(results.shape[0])])
    avg_fscore = np.average([rlresults[j][2] for j in range(results.shape[0])])
    avg_cossim = np.average(cossim_results)

    return [avg_prec, avg_recall, avg_fscore, avg_cossim]


def eval_against_humangenerated(method, embedding=None):
    """
    Compute ROUGE-L and cosine similarity evaluation measures for first five
    records where ideal summaries are human generated.

    :param method: LSA or TextRank summarization method.
    :param embedding: fastText word embedding.
    :return results: List containing evalution measure computations.
    :rtype: list(array_type): float
    """
    human_summaries = [
    ("It's never happened before in software! Remember, the "
     "hard part is not deciding what features to add, it's "
     "The lesson was: simplicity sells."),
    ("This is where I realized that there was really a need to communicate, "
     "because the data of what's happening in the world and the child "
     "health of every country is very well aware."
     "Now, statisticians don't like it, because they say that this will not "
     "show the reality; we have to have statistical, analytical methods. "
     "And it's a new technology coming in, but then amazingly, how well it "
     "fits to the economy of the countries."),
    ("And the interesting thing is: if you do it for love, the money comes "
     "anyway. 'To be successful, put your nose down in something and get "
     "damn good at it.' Persistence is the number one reason for our success."),
    ("So honeybees are important for their role in the economy as well as "
     "in agriculture. We need bees for the future of our cities and urban "
     "living. What can you do to save the bees or to help them or to think "
     "of sustainable cities in the future?"),
    ("So now I want to introduce you to my new hero in the global climate "
     "change war, and that is the eastern oyster. So the oyster was the "
     "basis for a manifesto-like urban design project that I did about the "
     "New York Harbor called oyster-tecture. To conclude, this is just one "
     "cross-section of one piece of city, but my dream is, my hope is, that "
     "when you all go back to your own cities that we can start to work "
     "together and collaborate on remaking and reforming a new urban "
     "landscape towards a more sustainable, a more livable and a more "
     "delicious future.")]
    df = read_data(5)
    results = []
    for j in range(len(df)):
        s = method.summarize_text(df.iloc[[j]], 3)
        rl = Evaluation.rouge_l(s, human_summaries[j])
        cs = Evaluation.cos_similarity(s, human_summaries[j])
        results.append([rl, cs])
    return results


def avg_results(results):
    """
    Compute average of results struct.

    :param resuls: List of evaluation measures for data records.
    :return: List of evaluation measure averages over the number of records.
    :rtype: list(float)
    """
    results = np.asarray(results)
    rlresults = results[:,0]
    cossim_results = results[:,1]
    avg_prec = np.average([rlresults[j][0] for j in range(results.shape[0])])
    avg_recall = np.average([rlresults[j][1] for j in range(results.shape[0])])
    avg_fscore = np.average([rlresults[j][2] for j in range(results.shape[0])])
    avg_cossim = np.average(cossim_results)
    return [avg_prec, avg_recall, avg_fscore, avg_cossim]


def score_all_data(df, summary_length, metric=rouge_n, N=2,
                   use_tag_concepts=True, embedding=None):
    """
    Compute evaluation measures for every record in dataframe.

    :param df: Dataframe containing trascripts, tags, and descriptions.
    :param summary_length: Desired length of returned text summary.
    :param metric: Evaluation metric to be computed.
    :param N: Parameter for ROUGE-N computuation (Evaluation.rouge_n).
    :param use_tag_concepts: Bool to signal using tags as document concepts.
    :param embedding: fastText word embedding.
    :return: Nunpy array of evaluation results.
    :rtype: array_type
    """
    results = np.zeros(len(df))
    for j in range(len(df)):
        s = summarize(df.iloc[[j]], summary_length, use_tag_concepts, embedding)
        ideal = preprocess_transcripts(df.iloc[[j]], df_field='description',
                                       clean_transcript_data=False)
        score = Evaluation.metric(s, ideal[0][0])
        results[j] = score
    return results


if __name__ == '__main__':
    df = read_data()
    df = df[::5]
    df = drop_noconcept_recs(df[:200])
    wvs = load_wordvectors()

    # ROUGE-1
    gtc7_r1 = score_all_data(df, 7, 1)
    gnt7_r1 = score_all_data(df, 7, 1, False)
    ctc7_r1 = score_all_data(df, 7, 1)
    cnt7_r1 = score_all_data(df, 7, 1, False)

    # ROUGE-2
    gtc7_r2 = score_all_data(df, 7, 2)
    gnt7_r2 = score_all_data(df, 7, 2, False)
    ctc7_r2 = score_all_data(df, 7, 2)
    cnt7_r2 = score_all_data(df, 7, 2, False)

    # Cosine similarity
    gtc7_cs = score_all_data(df, 7, cos_similarity, wvs)
    gnt7_cs = score_all_data(df, 7, cos_similarity, False, wvs)
    ctc7_cs = score_all_data(df, 7, cos_similarity, wvs)
    cnt7_cs = score_all_data(df, 7, cos_similarity, False, wvs)

    # TextRank
    txrk_r1 = score_all_data(df, 7, 1, wvs)
    txrk_r2 = score_all_data(df, 7, 2, wvs)
    txrk_cs = score_all_data(df, 7, cos_similarity, wvs)
