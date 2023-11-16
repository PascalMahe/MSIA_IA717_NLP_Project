from collections import Counter
import logging
from logging.handlers import TimedRotatingFileHandler
from math import log
import os
import string
from string import punctuation
from string import Template
import copy

from itertools import combinations
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from wordfreq import zipf_frequency

# if punkt is missing we download all packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')


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


def preprocess_sentence(sentence):
    stemmer = PorterStemmer()
    # tokenize and lower case
    tokens = word_tokenize(sentence.lower())
    # stemming and removing stop words
    stemmed_tokens = [stemmer.stem(token) for token in tokens if not token in set(
        stopwords.words('english'))]
    # remove punctuations
    filtered_tokens = [
        token for token in stemmed_tokens if token not in punctuation]
    # return ' '.join(filtered_tokens) # Anaele supprimé ici et intégré dans preprocess_dataset()
    return filtered_tokens  # renvoie une liste (vecteur de la phrase)


def preprocess_dataset(dataset):
    df = pd.DataFrame(dataset)

    df[['s1', 's2']] = df['data'].apply(lambda x: pd.Series(x))
    df['sm'] = df['s1'] + " " + df['s2']  # merged sentences
    # preprocessing des phrases 1 et 2 en vecteur ou string :
    df['token_1'] = df['s1'].apply(preprocess_sentence)  # vecteur
    df['token_2'] = df['s2'].apply(preprocess_sentence)  # vecteur
    df['token_m'] = df['sm'].apply(preprocess_sentence)  # vecteur
    df['s1_pp'] = df['token_1'].apply(' '.join)  # string
    df['s2_pp'] = df['token_2'].apply(' '.join)  # string
    df['sm_pp'] = df['token_m'].apply(' '.join)  # string
    df['scores_norm'] = df['scores'] / 5  # normalisation du score entre 0 et 1
    # Replace empty strings with NaN
    df.replace('', np.nan, inplace=True)

    # Store the index labels before dropna
    index_before_dropna = df.index

    # Drop rows with NaN values
    df.dropna(inplace=True)

    df.reset_index(drop=True)

    # Store the index labels after dropna
    index_after_dropna = df.index

    # Get the index labels that were dropped
    dropped_indexes = index_before_dropna.difference(index_after_dropna)

    logger.info("Indices dropped:")
    logger.info(dropped_indexes)

    return df

# helper function that gets the corpus, the words, the vocabulary (unique words)


def get_corpus(preprocessed_train):
    # récupère toutes les phrases du corpus (format string)
    corpus = preprocessed_train['s1_pp'].tolist(
    ) + preprocessed_train['s2_pp'].tolist()
    return corpus


def get_words(array):
    # récupère tous les mots (avec redondance)
    words = []  # liste de mots
    for word_list in array:
        if type(word_list) == str:
            # 'ne marche pas pour les string ?!?
            word_list = word_tokenize(word_list)
        words += word_list
    return words


def get_vocabulary(array):
    # récupère le vocabulaire unique (sans redondance)
    return np.unique(get_words(array))


def similarity(a: float, b: float) -> float:
    return 1 - abs(a - b) / max(abs(a), abs(b), 1)


def vector_similarity(vec_a, vec_b):
    # Calculate the dot product of the vectors
    dot_product = np.dot(vec_a, vec_b)

    # Calculate the L2 norm (Euclidean norm) of each vector
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # Calculate cosine similarity
    similarity = dot_product / (norm_a * norm_b)

    return similarity


def logit(p):
    if p <= 0:
        return 0
    elif p >= 1:
        return 1
    elif p > 0 and p < 1:
        return np.log(p / (1 - p))
    else:
        raise ValueError(
            "The input value must be in the closed interval [0, 1].")


def min_max_normalization_series(series):
    min_val = series.min()
    max_val = series.max()

    normalized_series = (series - min_val) / (max_val - min_val)
    return normalized_series


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

def show_examples(df, initial, feature, lst=(1, 2, 10, 22)):
    X = df[str(initial)]
    Y = df[str(feature)]
    error = X-Y
    for i in lst:
        vocabulary_i = get_vocabulary(df['token_m'][i])
        print('\n', i, ":", df['data'][i],
              '\n vocabulary:', vocabulary_i,
              '\n s1 :', df['s1_pp'][i], ' => ', df['token_1'][i],
              '\n s2 :', df['s2_pp'][i], ' => ', df['token_2'][i],
              '\n initial score "'+str(initial)+'" :', np.round(X[i], 2),
              '\n feature score "'+str(feature)+'" :', np.round(Y[i], 2),
              '\n longueur s1 & s2 :', len(df['token_1'][i]), ',', len(
                  df['token_2'][i]), "dif =", len(df['token_1'][i])-len(df['token_2'][i]),
              '\n error  :', np.round(error[i], 2))
        if Y[i] > 1:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',
                  '\n!!!     ATTENTION SCORE =', Y[i], ' (> 1)     !!!',
                  '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

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

def get_top_error(lst, n=3):
    lst = np.array(lst)
    top = sorted(np.unique(lst)[-3:], reverse=True)
    print('top 3 des erreurs:', np.round(top, 3))
    top_index = [(np.where(lst == top[0])[0])[0],
                 (np.where(lst == top[1])[0])[0],
                 (np.where(lst == top[2])[0])[0]]
    print('indices des top 3:', top_index, '\n')
    return top_index


def show_top_error(df, initial, feature):
    temp = df[['token_1', 'token_2', str(initial), str(feature)]].copy()
    temp['error'] = temp.iloc[:, 2]-temp.iloc[:, 3]
    top_error_index = get_top_error(temp['error'])
    return temp.iloc[top_error_index, :]

# the function given to compute the jaccard_score score doesn't work as is for the dataframe
# it's (barely) rewritten here to make it work
def feature0_scores(df):
    x = []
    for i in df.index:
        s1 = df["s1_pp"][i]
        try:
            s1 = df["s1_pp"][i]
        # s1 = df['s1'][i]
        except:
            logger.info(f"error while trying to access: df[\"s1_pp\"][{i}]")
            logger.info(f"before: {i-1}, Valeur: {df['s1'][i-1]}")
            logger.info(f"df[\"s1_pp\"].length: {len(df['s1_pp'])}")

            exit()
        s2 = df["s2_pp"][i]
        if len(s1) or len(s2):
            jacc_score = 0
        else:
            # binary=True because we use Jaccard score (we want presence/absence information, not counts)
            cv = CountVectorizer(binary=True)
            vectors = cv.fit_transform([s1, s2]).toarray()
            jacc_score = jaccard_score(vectors[0], vectors[1])
        x.append(jacc_score)

    df["scores_0"] = x
    return df


def feature1_scores(df, count_w=True):
    # if count_w=False will return a vector with number of occurence
    # if count_w=True will return a vector with 1 for words present / 0 for words absent (Jaccard score)

    for i in df.index:
        vocabulary_i = get_vocabulary(df['token_m'][i])
        # binary=True ne marche pas ?!?
        vectorizer = CountVectorizer(vocabulary=vocabulary_i, binary=count_w)
        v_t1 = vectorizer.fit_transform(df['token_1'][i]).toarray().sum(axis=0)
        v_t2 = vectorizer.fit_transform(df['token_2'][i]).toarray().sum(axis=0)
        # d_eucl_linalg = np.linalg.norm(v_t1-v_t2)

        d_eucl = euclidean(v_t1, v_t2)  # scipy.spatial.distance.euclidian()

        # if v_t1 or v_t2 are all zeros, there's a divide by 0 error in cosine()
        all_zeros_1 = np.all(v_t1 == 0)
        all_zeros_2 = np.all(v_t2 == 0)
        if not all_zeros_1 and not all_zeros_2:
            # scipy.spatial.distance.cosine()
            d_cos = 1 - cosine(v_t1, v_t2)
        else:
            d_cos = 0

        np.seterr(all="warn")
        # df.loc[i, 'scores_1_eucl'] = d_eucl_linalg
        df.loc[i, 'scores_1_eucl'] = d_eucl
        df.loc[i, 'scores_1_cos'] = d_cos
    return df


def build_frequency_vector(vocab_as_list, sentence):
    frequency_vectors_sentence = np.zeros(len(vocab_as_list))
    for token in sentence:
        token_index = vocab_as_list.index(token)
        frequency_vectors_sentence[token_index] = 1
    return frequency_vectors_sentence


def find_synonyms(word):
    return [syn for synset in wordnet.synonyms(word) for syn in synset]


def feature2_scores(df):
    df['scores_2'] = 0
    for i in df.index:
        s1 = df['s1_pp'][i]
        s2 = df['s2_pp'][i]

        for j, token_1 in enumerate(s1.split()):
            s1_syn = wordnet.synsets(token_1)
            if (token_1 in s2.split()):
                df.loc[i, 'scores_2'] += 1
                continue
            for k, token_2 in enumerate(s2.split()):
                s1_syn = set(find_synonyms(token_1))
                for syn in s1_syn:
                    if (syn == token_2):
                        df.loc[i, 'scores_2'] += 1

    return df


def feature3_scores(df):

    # building vocabulary out of every sentence
    corpus = []
    for i in df.index:
        # TfidfVectorizer expects the corpus to be list of strings
        # so the token lists are joined back as strings before
        # being added to the corpus
        corpus.append(' '.join(df["token_1"][i]))
        corpus.append(' '.join(df["token_2"][i]))

    # TFIDF vectorizer
    vectorizer = TfidfVectorizer()
    # train it on whole corpus
    vectorizer.fit_transform(corpus)

    for i in df.index:
        sentence1 = ' '.join(df["token_1"][i])
        sentence2 = ' '.join(df["token_2"][i])

        sentence1_tfidf = vectorizer.transform([sentence1])
        sentence2_tfidf = vectorizer.transform([sentence2])

        # compute similarity score
        cosine_similarity_score = cosine_similarity(
            sentence1_tfidf, sentence2_tfidf)[0][0]

        df.loc[i, 'scores_3'] = cosine_similarity_score
    return df

# First of all, I need some helper functions
# Especially helpful for feature 5 code


def find_synonyms(word):
    return [syn for synset in wordnet.synonyms(word) for syn in synset]


def replace_with_synonyms(s1, s2):
    s1_tokens = s1.split()
    s2_tokens = s2.split()

    # Replace tokens in s2 with synonyms from s1
    replaced_s2_tokens = []
    for token in s2_tokens:
        syn_found = False
        synonyms_list = find_synonyms(token)
        for syn in synonyms_list:
            # logger.info(syn)
            if syn.lower() in s1_tokens:
                replaced_s2_tokens.append(syn.lower())
                syn_found = True
                break
        if syn_found == False:
            replaced_s2_tokens.append(token)

    return ' '.join(replaced_s2_tokens)


def n_gram_overlap(df, corpus, synonym_based=False):
    # Create a CountVectorizer with different N-gram values (1, 2, and 3)
    vectorizers = [CountVectorizer(ngram_range=(1, 1)),
                   CountVectorizer(ngram_range=(1, 2)),
                   CountVectorizer(ngram_range=(1, 3))]

    for i, vectorizer in enumerate(vectorizers):
        # Fit the vectorizer with the corpus
        vectorizer.fit(corpus)

        cosines = []
        euclidians = []

        for j in df.index:
            s1, s2 = df['s1_pp'][j], df['s2_pp'][j]
            if synonym_based:
                s2 = replace_with_synonyms(s1, s2)
            # Calculate Euclidean distance
            euclidean_distance = euclidean_distances(
                vectorizer.transform([s1]), vectorizer.transform([s2]))
            euclidians.append(euclidean_distance[0][0])
            # Calculate cosine similarity
            cosine = cosine_similarity(vectorizer.transform(
                [s1]), vectorizer.transform([s2]))
            cosines.append(cosine[0][0])

        # Create new columns in the data frame
        if synonym_based:
            df[f'scores_5_euclidean_{i+1}_syn'] = euclidians
            df[f'scores_5_cosine_{i+1}_syn'] = cosines
        else:
            df[f'scores_4_euclidean_{i+1}'] = euclidians
            df[f'scores_4_cosine_{i+1}'] = cosines

    return df


def similarity(a: float, b: float) -> float:
    return 1 - abs(a - b) / max(abs(a), abs(b), 1)


def cosine_vector_similarity(vec_a, vec_b):
    # Calculate the dot product of the vectors
    dot_product = np.dot(vec_a, vec_b)

    # Calculate the L2 norm (Euclidean norm) of each vector
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # Calculate cosine similarity
    similarity = dot_product / (norm_a * norm_b)

    return similarity


def logit(p):
    if p <= 0:
        return 0
    elif p >= 1:
        return 1
    elif p > 0 and p < 1:
        return np.log(p / (1 - p))
    else:
        raise ValueError(
            "The input value must be in the closed interval [0, 1].")


def min_max_normalization_series(series):
    min_val = series.min()
    max_val = series.max()

    normalized_series = (series - min_val) / (max_val - min_val)
    return normalized_series


def pos_synset_score(word_list):
    pos_score = 0
    synset_score = 0

    pos_tags = pos_tag(word_list)
    for word, tag in pos_tags:
        if tag.startswith('NN') or tag.startswith('VB'):
            pos_score += 2
        elif tag.startswith('RB'):
            pos_score += 1
        else:
            pos_score += 0.5

        synsets = wn.synsets(word)
        num_synsets = len(synsets)
        if num_synsets > 0:
            synset_score += 1 / num_synsets

    return [pos_score, synset_score]


def feature6_scores(preprocessed_data):

    pos_syn_1 = preprocessed_data['token_1'].apply(pos_synset_score)
    pos_syn_2 = preprocessed_data['token_2'].apply(pos_synset_score)
    preprocessed_data['scores_6'] = np.vectorize(
        cosine_vector_similarity)(pos_syn_1, pos_syn_2)
    # preprocessed_data['scores_6'] = np.vectorize(logit)(preprocessed_data['scores_6'])

    return preprocessed_data


def custom_tfidf(words):
    """This compute the tf idf score of each word in a list of words

    Args:
        words (_type_): a list of words

    Returns:
        _type_: a dictionary of words and their tf idf score
    """
    # Calculate term frequency (TF) for each word in the text
    tf = Counter(words)
    # Calculate "TF-IDF-like" score for each word
    tfidf_scores = {}
    for word, freq in tf.items():
        # Use Zipf frequency as a stand-in for IDF
        idf = 1 / (1+zipf_frequency(word, 'en'))

        # Calculate TF-IDF-like score
        tfidf_scores[word] = freq * idf

    return tfidf_scores


def compute_average_sentence_idf(sentence_list):

    # compute tfidf
    tfidfs = custom_tfidf(sentence_list)
    sum = 0
    for word, score in tfidfs.items():
        sum += score

    if (len(tfidfs) == 0):
        return 0

    return sum/len(tfidfs)


def feature7_scores(preprocessed_data):

    wordfreq_tfidf_1 = preprocessed_data['token_1'].apply(
        compute_average_sentence_idf)
    wordfreq_tfidf_2 = preprocessed_data['token_2'].apply(
        compute_average_sentence_idf)
    similarity_scores = np.vectorize(similarity)(
        wordfreq_tfidf_1, wordfreq_tfidf_2)

    preprocessed_data['scores_7'] = similarity_scores

    return preprocessed_data


debug_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s.%(funcName)s l.%(lineno)d | %(message)s')
debug_file_handler = TimedRotatingFileHandler(
    filename="features.log", when='midnight', backupCount=31)
debug_file_handler.setFormatter(debug_formatter)
debug_file_handler.setLevel(logging.DEBUG)
logger = logging.getLogger("features")
logger.setLevel(logging.DEBUG)
logger.addHandler(debug_file_handler)


dataset = load_data()
pp = preprocess_sentence('what a nice hat you have')
logger.info(f"%s : %s", pp, type(pp))
test = preprocess_dataset(dataset['test'])
preprocessed_train = preprocess_dataset(dataset['train'])
logger.info(f"Colonnes : %s", preprocessed_train.columns)
preprocessed_train.head()

corpus = get_corpus(preprocessed_train)
# logger.info(f"%s \n %s", len(corpus), corpus)

# test de get_words et get_vocabulary sur les premieres phrases s1
test = preprocessed_train['token_1'][0:5]
# logger.info(type(test))
logger.info(test)  # test.shape

words_test = get_words(test)
logger.info(f"\nwords: %s mots\n %s", len(words_test), words_test)
vocabulary_test = get_vocabulary(test)
logger.info(f"\nvocabulary (unique): %s mots uniques \n %s",
            len(vocabulary_test), vocabulary_test)


syns = wordnet.synsets("program")
logger.info(syns[0].name())  # First synonym


# features
# preprocessed_train = feature0_scores(preprocessed_train)
# preprocessed_train = feature1_scores(preprocessed_train)
# preprocessed_train = feature2_scores(preprocessed_train)
# preprocessed_train = feature3_scores(preprocessed_train)
# preprocessed_train = feature6_scores(preprocessed_train)
# preprocessed_train = feature7_scores(preprocessed_train)

# Extract features
def extract_features(dataset):
    preprocessed_dataset = preprocess_dataset(dataset)
    corpus = get_corpus(preprocessed_dataset)

    preprocessed_dataset = feature0_scores(preprocessed_dataset)
    preprocessed_dataset = feature1_scores(preprocessed_dataset)
    preprocessed_dataset = feature2_scores(preprocessed_dataset)
    preprocessed_dataset = feature3_scores(preprocessed_dataset)
    preprocessed_dataset = n_gram_overlap(
        preprocessed_dataset, corpus, synonym_based=False)
    preprocessed_dataset = n_gram_overlap(
        preprocessed_dataset, corpus, synonym_based=True)
    preprocessed_dataset = feature6_scores(preprocessed_dataset)
    preprocessed_dataset = feature7_scores(preprocessed_dataset)

    return preprocessed_dataset


def post_process_data(df):
    # select columns that need post-processing: those that start with 'scores_'
    # Ahmed here : I suggest to just drop the columns that we don't need by name
    # because i have many variants of scores and i can't align to this naming convention
    columns_to_normalize = []
    for i in range(7):
        current_column_name = "scores_" + str(i)
        if current_column_name in df.columns:
            columns_to_normalize.append(current_column_name)

    df[columns_to_normalize] = StandardScaler(
    ).fit_transform(df[columns_to_normalize])
    return df

# Build and train different models. You can do a little feature ablation (i.e. removing one feature at a time)
# to see the usefulness of the different features.


def test_models_with_feature_combinations(dataset, preprocess_function, postprocess_function, models):
    results = []

    preprocessed_train = preprocess_function(dataset['train'])
    postprocessed_train = postprocess_function(preprocessed_train)

    preprocessed_test = preprocess_function(dataset['test'])
    postprocessed_test = postprocess_function(preprocessed_test)

    # Fill NaN values if any
    postprocessed_train = postprocessed_train.fillna(0)
    postprocessed_test = postprocessed_test.fillna(0)

    # Extract features and targets
    train_features = [
        col for col in postprocessed_train.columns if col.startswith('scores_')]
    test_features = [
        col for col in postprocessed_test.columns if col.startswith('scores_')]
    score_to_predict = postprocessed_train['scores']
    score_to_verify = postprocessed_test['scores']

    logger.debug(f"score_to_predict.keys: %s", score_to_predict.keys())

    # Iterate over all combinations of features
    for r in range(1, len(train_features) + 1):
        logger.debug(f"r: %s", r)
        for feature_combination in combinations(train_features, r):
            logger.debug(
                f"feature_combination: %s", feature_combination)
            if len(feature_combination) > 3:
                raise RuntimeError("STOP, hammer time!")

            train_x = postprocessed_train[list(feature_combination)]
            test_x = postprocessed_test[list(feature_combination)]

            for model in models:
                logger.debug(
                    f"model: {model}, features: {feature_combination}")
                model_instance = model()
                logger.debug(f"train_x.shape: %s", train_x.shape)
                logger.debug(f"score_to_predict.shape: %s",
                             score_to_predict.shape)
                model_instance.fit(train_x, score_to_predict)
                predictions = model_instance.predict(test_x)

                mse = mean_squared_error(score_to_verify, predictions)
                r2 = r2_score(score_to_verify, predictions)

                results.append({
                    'features': feature_combination,
                    'model': model.__name__,
                    'MSE': mse,
                    'R2': r2
                })

    return pd.DataFrame(results)

# Showing results
show_scores(preprocessed_train, 'scores_norm', 'scores_0', 100)
show_top_error(preprocessed_train, 'scores_norm', 'scores_0')
examples = (1, 2, 10, 22)
show_examples(preprocessed_train, 'scores_norm', 'scores_0', examples)

show_scores(preprocessed_train, 'scores_norm', 'scores_1_eucl', 200)
show_top_error(preprocessed_train, 'scores_norm', 'scores_1_cos')
show_scores(preprocessed_train, 'scores_norm', 'scores_1_cos', 200)

show_scores(preprocessed_train, 'scores', 'scores_2', 200)

show_scores(preprocessed_train, 'scores_norm', 'scores_3', 200)

show_scores(preprocessed_train, 'scores_norm', 'scores_4_cosine_2', 100)
# affichage des score pour le 2-gram (disctance=euclidienne)
show_scores(preprocessed_train, 'scores', 'scores_4_euclidean_2', 100)

show_scores(preprocessed_train, 'scores_norm', 'scores_6', 100)

show_scores(preprocessed_train, 'scores_norm', 'scores_7', 100)


another_set = {
    'train': dataset['train'],
    'test': dataset['test']
}

models_to_test = [LinearRegression, Ridge, DecisionTreeRegressor]

logger.debug(f"Starting to combine model. Models: %s", models_to_test)

results_df = test_models_with_feature_combinations(
    another_set, extract_features, post_process_data, models_to_test)

results_df
