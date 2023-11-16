
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from string import punctuation

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
from nltk import pos_tag
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

try:
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    nltk.download('universal_tagset')
    nltk.download('averaged_perceptron_tagger')


debug_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s.%(funcName)s l.%(lineno)d | %(message)s')
debug_file_handler = TimedRotatingFileHandler(
    filename="co_occurrence_matrix.log", when='midnight', backupCount=31)
debug_file_handler.setFormatter(debug_formatter)
debug_file_handler.setLevel(logging.DEBUG)
logger = logging.getLogger("co_occurrence_matrix")
logger.setLevel(logging.DEBUG)
logger.addHandler(debug_file_handler)

# Building distributional representations
sws = stopwords.words('english')
sws = set(list(sws) + [p for p in punctuation])

def load_data():
    data = dict()
    for fn in os.listdir("stsbenchmark"):
        if fn.endswith(".csv"):
            with open("stsbenchmark/" + fn) as f:
                subset = fn[:-4].split("-")[1]
                logger.info(f"subset: %s", subset)
                data[subset] = dict()
                data[subset]['data'] = []
                data[subset]['scores'] = []
                for l in f:
                    # genre filename year score sentence1 sentence2 (and sources, sometimes)
                    l = l.strip().split("\t")
                    data[subset]['data'].append((l[5], l[6]))
                    data[subset]['scores'].append(float(l[4]))
    return data


def put_sentences_together(data):
    all_sentences = []
    for sentence_tuple in data:
        s1 = sentence_tuple[0]
        s2 = sentence_tuple[1]
        all_sentences.extend([s1, s2])
    return all_sentences


def create_vocabulary(data, count_threshold=1, voc_threshold=None, stopwords=set(), lowercase=False):
    """
    Function using word counts to build a vocabulary
    Params:
        data: list of tuples of sentences (strings)
        corpus (list of list of strings): corpus of sentences
        count_threshold (int): minimum number of occurences necessary for a word to be included in the vocabulary
        voc_threshold (int): maximum size of the vocabulary
        stopwords: a set of words which are excluded from the vocabulary
        lowercase: bool. If True, all words are lowercased (which results in a smaller, more compact vocabulary)
    IMPORTANT: the vocabulary includes "UNK", which is a placeholder for an unknown word and it will later be assigned a zero vector.
    Returns:
        vocabulary (dictionary): keys: list of distinct words across the corpus
                                 values: indexes corresponding to each word sorted by frequency
    """
    corpus = put_sentences_together(data)
    word_counts = {}
    for sent in corpus:
        for word in word_tokenize(sent):
            if lowercase:
                word = word.lower()
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1

    # Create a dictionary called <filtered_word_counts> (with words as keys and their frequencies as values).
    # Include only those words that appear more than <count_threshold> times,
    # and which are not in the set of stopwords.
    filtered_word_counts = {word: freq for word,
                            freq in word_counts.items() if word not in stopwords and freq >= count_threshold}

    # Create a list called <words> sorting the words from highest to lowest frequency
    # 1st step: sorting filtered_word_count by frequency (see https://stackoverflow.com/a/613218/2112089)
    filtered_word_counts = dict(
        sorted(filtered_word_counts.items(), key=lambda item: item[1], reverse=True))
    # 2nd step: extract the words as a list
    words = list(filtered_word_counts.keys())

    if voc_threshold is not None:
        words = words[:voc_threshold] + ['UNK']
    vocabulary = {words[i]: i for i in range(len(words))}
    return vocabulary, {word: filtered_word_counts.get(word, 0) for word in vocabulary}


def co_occurence_matrix(dataset, vocab, lowercase=False):
    """
    Params:
        dataset: output of load_data()
        vocab: first output of create_vocabulary(). These are the words that will be included in the matrix
    Returns:
        matrix (array of size (len(vocab), len(vocab))): the co-occurrence matrix, using the same ordering as the vocabulary given in input
    """
    l = len(vocab)
    all_sentences = put_sentences_together(dataset)
    M = np.zeros((l, l))
    for sent in all_sentences:
        sent_idcs = []
        for word in word_tokenize(sent):
            if lowercase:
                word = word.lower()
            sent_idcs.append(vocab.get(word, len(vocab)-1))
        for i, idx in enumerate(sent_idcs):
            for j, ctx_idx in enumerate(sent_idcs[i+1:]):
                M[idx][ctx_idx] += 1
                M[ctx_idx][idx] += 1
    return M


def pmi(M, positive=True):
    """A function that converts the matrix values to PMI"""
    sum_vec = M.sum(axis=0)
    sum_tot = sum_vec.sum()
    with np.errstate(divide='ignore'):
        pmi = np.log((M * sum_tot) / (np.outer(sum_vec, sum_vec)))
    pmi[np.isinf(pmi)] = 0.0  # log(0) = 0
    if positive:
        pmi[pmi < 0] = 0.0
    return pmi


def extract_token_repres_from_co_oc_matrix(token, vocab, co_occurrence_matrix):
    # for some reason, vocab is a dictionary where keys are tokens and values are the token's index
    # in the dict
    # (this feels like it should be a list, and the function .index(token) used to get the index)
    token_index_in_vocab = vocab[token]
    token_repres = co_occurrence_matrix[token_index_in_vocab]
    return token_repres


def build_sentence_representation(tokened_sen, vocab, co_occurrence_matrix, postag=False, possed_sen=None):
    sen_repres = []
    considered_tokens = []
    if postag:
        # if postag, then we check if the pos of the token is a verb, noun, or adjective
        # otherwise it's ignored
        for token, pos in possed_sen:
            if (pos.startswith("NN") or pos.startswith("VB") or pos.startswith("ADV")) and token in vocab:
                sen_repres.append(extract_token_repres_from_co_oc_matrix(
                    token, vocab, co_occurrence_matrix))
                considered_tokens.append(token)
    else:
        for token in tokened_sen:
            # words not in the vocab are ignored
            if token in vocab:
                # if the word is in the vocab, we get its representation from the M matrix
                # said representation is the nth row, whre n is the index in the vocab
                sen_repres.append(extract_token_repres_from_co_oc_matrix(
                    token, vocab, co_occurrence_matrix))
                considered_tokens.append(token)
    # logger.debug(f"considered_tokens: %s", considered_tokens)
    return sen_repres


def assign_distributional_vectors(data, M, vocab, sim_or_dist=True, postag=False, lowercase=False):
    '''This functions assigns each sentence a vector and optionally calculates the similarity/distance
    between the representations of s1 and s2.
    Parameters
      data: list of tuples (like dataset['train']['data'])
      M: a matrix of distributional representations for all words in the vocabulary
      vocab: first output of create_vocabulary(). These are the words that will be included in the matrix
      sim_or_dist: bool. If True, we will use a similarity or distance as the only feature. If False,
      we will use the concatenation of the representations of s1 and s2.
      postag: whether we want to apply a postag-based filter to obtain sentence representations
      lowercase: bool. If True, words are lowercased. You should set it to True if the vocabulary is lowercased.
    Returns:
      features: an array with the data transformed into features '''
    if sim_or_dist:
        features = []
    else:
        features = np.zeros((len(data), M.shape[1]*2))
    # logger.debug(f"features.shape: %s", features.shape)
    for i, (s1, s2) in enumerate(data):

        # Tokenize, lowercase if lowercase=True, and if postag=True, postag s1 and s2
        if lowercase:
            s1 = s1.lower()
            s2 = s2.lower()

        tokened_s1 = word_tokenize(s1)
        tokened_s2 = word_tokenize(s2)

        possed_s1 = None
        possed_s2 = None
        if postag:
            possed_s1 = pos_tag(tokened_s1)
            possed_s2 = pos_tag(tokened_s2)

        # Now create two lists, one for each sentence, with the word representations that you want to use
        # You can go through the words (or word, pos) in each sentence and decide whether you keep
        # their representation or not
        s1vecs = build_sentence_representation(
            tokened_s1, vocab, M, postag, possed_s1)
        s2vecs = build_sentence_representation(
            tokened_s2, vocab, M, postag, possed_s2)

        # It is possible that some sentences will not have any word representation available.
        # We assign them a 0-vector in this case (be careful, because this could result in a cosine of NaN)
        if not s1vecs:
            s1vecs = [np.zeros(M.shape[1])]
        if not s2vecs:
            s2vecs = [np.zeros(M.shape[1])]

        # Aggregate the representations of words in a sentence, for example by averaging them
        agg_s1 = np.array(s1vecs).mean(0)
        agg_s2 = np.array(s2vecs).mean(0)

        # Fill in features[i] with the desired feature (one or more similarity/distance measures if sim=True,
        # a concatenation of the representations otherwise)
        if sim_or_dist:
            # cosine distance meaning we substract from 1
            # scipy.spatial.distance.cosine()
            value_to_set = 1 - cosine(agg_s1, agg_s2)
            # if value_to_set != 1:
            #     logger.debug(
            #         f"value_to_set is not 1 for sentence group %s, it's %s", i, value_to_set)
            features.append(value_to_set)
        else:
            # concatenation of the repr of s1 and s2
            value_to_set = np.concatenate((agg_s1, agg_s2))
            features[i] = value_to_set
    # logger.debug(f"features.length: %s", len(features))
    if sim_or_dist:
        features = np.array(features)
    return features

def MSE(X, Y):
    if len(X) == len(Y):
        npX = np.array(X)
        npY = np.array(Y)
        MSE = ((npY - npX.mean())**2).sum()/len(npX)
    else:
        "les deux vecteurs fournis doivent etre de meme dimension pour calculer le MSE"
    return MSE

def coeff_colelation(X, Y):
    if len(X) == len(Y):
        npX = np.array(X)
        npY = np.array(Y)
        cov = ((npX - npX.mean())*(npY - npY.mean())).sum()/len(npX)
    else:
        "les deux vecteurs fournis doivent etre de meme dimension pour calculer le coefficient de corrélation"
    coeff = cov / (npX.std()*npY.std())
    return coeff

def show_stats(df, initial, feature):
    X = df[str(initial)]
    Y = df[str(feature)]
    error = X-Y
    print('Le score moyen (initial) est de :', round(
        X.mean(), 2), '(écart type de', round(X.std(), 2), ')')
    print('Le score moyen de la feature est de :', round(
        Y.mean(), 2), '(écart type de', round(Y.std(), 2), ')')
    print('L\'erreur est en moyenne de :', round(error.mean(), 2),
          '(écart type de', round(error.std(), 2), ')')
    print('Le coefficient de corélation de "', str(initial), '" et "',
          str(feature), '" est :', round(coeff_colelation(X, Y), 2))
    print('Le MSE est de :', round(MSE(X, Y), 2))

def show_scores(df, initial, feature, number, start=-1):
    # permet de choisir les données à afficher
    X = df[str(initial)]
    Y = df[str(feature)]
    error = X-Y

   # choix de l'échantillon à afficher
    selection = np.array([])
    if start == -1:
        selection = np.random.randint(0, len(X), number)
        selection.sort()
    elif start >= 0:
        selection = np.array(range(start, start+number))
    else:
        print('start doit etre égal à -1 (aléatoire) ou etre >=0 pour indiquer l\'index à partir duquel afficher les points')
    X_show = X[selection]
    Y_show = Y[selection]
    error_show = error[selection]

    # paramétrage de l'abcisse
    x_min = min(selection)
    x_max = max(selection)
    if start == -1:
        selection_type = ' comparaisons aléatoires'
    else:
        selection_type = ' comparaisons à partir de l\'indice'+str(start)
    y_max = max(np.ceil(max(X)), np.ceil(max(Y)))

    # indication des stats sur les scores et l'erreur
    show_stats(df, initial, feature)

    # affichage du graphique
    plt.plot(X_show, label='Score')
    plt.plot(Y_show, label='feature score')
    plt.plot(error_show, ".", label='error', color='grey')
    # plt.plot(initial.head(number), label = 'Score')
    # plt.plot(feature.head(number), label = 'feature score')
    # plt.plot(error.head(number), ".", label = 'error', color = 'grey')
    plt.xlabel(str(number) + selection_type, fontsize=10)
    plt.ylabel('Score (entre 0 et ' + str(y_max)+')', fontsize=10)
    plt.title('\nComparaison du score de \"' + str(feature) +
              '\" \nau score initial "' + str(initial)+"'", fontsize=14)
    plt.axis(ymax=y_max)
    plt.plot((x_min, x_max), (0, 0), color='k')
    plt.plot((x_min, x_max), (max(X), max(X)), color='k')
    plt.plot((x_min, x_max), (error.mean(), error.mean()),
             color='grey', linestyle=':', label='erreur moyenne')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()

dataset = load_data()

# slicing the dataset to keep only a human-manageable number of sentences
training_data = dataset['train']['data']
# logger.info(f"dataset.length: %s", len(training_data))
# debug_training_data = random.choices(training_data, k=20)
# training_data = training_data[:20]  # debug version
logger.info(f"training_data.length: %s", len(training_data))
# logger.info(f"debug_dataset: %s", training_data)

# Build vocabulary
vocab, word_freq = create_vocabulary(training_data, count_threshold=2,
                                     voc_threshold=None, stopwords=sws, lowercase=True)
logger.debug(f"word_frequency.length: %s", len(word_freq))
# logger.debug(f"word_frequency: %s", word_freq)

# # Build co-occurrence matrix
M = co_occurence_matrix(training_data, vocab)
# logger.debug(f"M.shape: %s", M.shape)
# logger.debug(f"M:\n%s", M)

# To apply the transformation:
PMI_M = pmi(M)
# logger.debug(f"PMI_M.shape: %s", PMI_M.shape)
# logger.debug(f"PMI_M:\n%s", PMI_M)

# because we have 3 bool parameters, we can get 8 features out of this single function
# 4 first: concatenation
# sim_or_dist = False
# postag = False
# lowercase = False
# feature_conc_no_pos_no_lower = assign_distributional_vectors(
#     training_data, PMI_M, vocab, sim_or_dist, postag, lowercase)

# sim_or_dist = False
# postag = False
# lowercase = True
# feature_conc_no_pos_yes_lower = assign_distributional_vectors(
#     training_data, M, vocab, sim_or_dist, postag, lowercase)

# sim_or_dist = False
# postag = True
# lowercase = False
# feature_conc_yes_pos_no_lower = assign_distributional_vectors(
#     training_data, M, vocab, sim_or_dist, postag, lowercase)

# sim_or_dist = False
# postag = True
# lowercase = True
# feature_conc_yes_pos_yes_lower = assign_distributional_vectors(
#     training_data, M, vocab, sim_or_dist, postag, lowercase)

# logger.debug(f"feature_conc_no_pos_no_lower:\n%s",
#              feature_conc_no_pos_no_lower)
# logger.debug(f"feature_conc_no_pos_yes_lower:\n%s",
#              feature_conc_no_pos_yes_lower)
# logger.debug(f"feature_conc_yes_pos_no_lower:\n%s",
#              feature_conc_yes_pos_no_lower)
# logger.debug(f"feature_conc_yes_pos_yes_lower:\n%s",
#              feature_conc_yes_pos_yes_lower)

# # 4 last: similarity
sim_or_dist = True
postag = False
lowercase = False
feature_sim_no_pos_no_lower = assign_distributional_vectors(
    training_data, M, vocab, sim_or_dist, postag, lowercase)

sim_or_dist = True
postag = False
lowercase = True
feature_sim_no_pos_yes_lower = assign_distributional_vectors(
    training_data, M, vocab, sim_or_dist, postag, lowercase)

sim_or_dist = True
postag = True
lowercase = False
feature_sim_yes_pos_no_lower = assign_distributional_vectors(
    training_data, M, vocab, sim_or_dist, postag, lowercase)

sim_or_dist = True
postag = True
lowercase = True
feature_sim_yes_pos_yes_lower = assign_distributional_vectors(
    training_data, M, vocab, sim_or_dist, postag, lowercase)


# logger.debug(f"feature_sim_no_pos_no_lower:\n%s",
#              feature_sim_no_pos_no_lower)
# logger.debug(f"feature_sim_no_pos_no_lower.shape: %s",
#              feature_sim_no_pos_no_lower.shape)
# logger.debug(f"feature_sim_no_pos_no_lower[0].shape: %s",
#              feature_sim_no_pos_no_lower[0].shape)
# logger.debug(f"feature_sim_no_pos_no_lower[0][0].shape: %s",
#              feature_sim_no_pos_no_lower[0][0].shape)
# logger.debug(f"feature_sim_no_pos_no_lower[0][0].type: %s",
#              type(feature_sim_no_pos_no_lower[0][0]))
# logger.debug(f"feature_sim_no_pos_yes_lower:\n%s",
#              feature_sim_no_pos_yes_lower)
# logger.debug(f"feature_sim_no_pos_yes_lower[0].shape: %s",
#              feature_sim_no_pos_yes_lower[0].shape)
# logger.debug(f"feature_sim_yes_pos_no_lower:\n%s",
#              feature_sim_yes_pos_no_lower)
# logger.debug(f"feature_sim_yes_pos_no_lower[0].shape: %s",
#              feature_sim_yes_pos_no_lower[0].shape)
# logger.debug(f"feature_sim_yes_pos_yes_lower:\n%s",
#              feature_sim_yes_pos_yes_lower)
# logger.debug(f"feature_sim_yes_pos_yes_lower[0].shape: %s",
#              feature_sim_yes_pos_yes_lower[0].shape)

scores_to_verify = np.array(dataset['test']['scores'])[
    :20] / 5  # debug version
scores_to_verify = np.array(dataset['test']['scores'])[:20] / 5

linreg = LinearRegression()
# Training
# 1) computing features
# now we need to create an array from the features
# at the moment, they're (n,) meaning they're 1D arrays
# (see: https://stackoverflow.com/questions/34932739/python-numpy-shape-confusion)
# we need them to make a 2D array, to have numpy.hstack to behave properly
# they need to be (n,1), to do that, we use: .reshape(-1, 1) (see: idem)
training_features = np.hstack(
    [feature_sim_no_pos_no_lower.reshape(-1, 1),
     feature_sim_no_pos_yes_lower.reshape(-1, 1),
     feature_sim_yes_pos_no_lower.reshape(-1, 1),
     feature_sim_yes_pos_yes_lower.reshape(-1, 1)])
# training_features = np.array(feature_sim_no_pos_no_lower)
logger.debug(f"training_features.shape: %s", training_features.shape)

# scaling scores from (0, 5) to (0, 1)
# scores_to_predict = np.array(dataset['train']['scores'])[
#     :20] / 5  # debug version
scores_to_predict = np.array(dataset['train']['scores']) / 5

# 2) fitting model
linreg.fit(training_features, scores_to_predict)

# Predicting
# 1) computing features on test data
# testing_data = np.array(dataset['test']['data'][:20])  # debug version
testing_data = np.array(dataset['test']['data'])
logger.debug(f"testing_data.shape: %s", testing_data.shape)

sim_or_dist = True
postag = False
lowercase = False
testing_sim_no_pos_no_lower = assign_distributional_vectors(
    testing_data, M, vocab, sim_or_dist, postag, lowercase)

sim_or_dist = True
postag = False
lowercase = True
testing_sim_no_pos_yes_lower = assign_distributional_vectors(
    testing_data, M, vocab, sim_or_dist, postag, lowercase)

sim_or_dist = True
postag = True
lowercase = False
testing_sim_yes_pos_no_lower = assign_distributional_vectors(
    testing_data, M, vocab, sim_or_dist, postag, lowercase)

sim_or_dist = True
postag = True
lowercase = True
testing_sim_yes_pos_yes_lower = assign_distributional_vectors(
    testing_data, M, vocab, sim_or_dist, postag, lowercase)

testing_features = np.hstack(
    [testing_sim_no_pos_no_lower.reshape(-1, 1),
     testing_sim_no_pos_yes_lower.reshape(-1, 1),
     testing_sim_yes_pos_no_lower.reshape(-1, 1),
     testing_sim_yes_pos_yes_lower.reshape(-1, 1)])

# 2) predicting with model
predictions = linreg.predict(testing_features)

validation_scores = np.array(dataset['test']['scores']) / 5

df_train = pd.DataFrame(dataset['train'])


logger.debug(f"predictions.shape: %s", predictions.shape)
logger.debug(f"predictions: %s", predictions)
logger.debug(f"validation_scores.shape: %s", validation_scores.shape)
logger.debug(f"validation_scores: %s", validation_scores)
plt.plot(validation_scores)
plt.show()

logger.info("FINISHED")
print("FINISHED")
