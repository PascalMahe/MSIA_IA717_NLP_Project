from collections import Counter
import os
from string import punctuation

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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from wordfreq import zipf_frequency
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
from logger import logger
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from rouge_score import rouge_scorer
import statsmodels.api as sm

# if punkt is missing we download all packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

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


def preprocess_sentence_no_stem(sentence):
    # tokenize and lower case
    tokens = word_tokenize(sentence.lower())
    # stemming and removing stop words
    stemmed_tokens = [token for token in tokens if not token in set(
        stopwords.words('english'))]
    
        
    for token in stemmed_tokens:
        if token == "n t":
            token="not"
        elif token == "m":
            token = "am"
    # remove punctuations
    filtered_tokens = [token for token in stemmed_tokens if token not in punctuation]
    
    return ' '.join(filtered_tokens)

def preprocess_sentence(sentence):
    stemmer = PorterStemmer()
    # tokenize and lower case
    tokens = word_tokenize(sentence.lower())
    # stemming and removing stop words
    stemmed_tokens = [stemmer.stem(token) for token in tokens if not token in set(
        stopwords.words('english'))]
    
    for token in stemmed_tokens:
        if token == "n t":
            token="not"
        elif token == "m":
           token = "am"

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
    df['nostem_1'] = df['s1'].apply(preprocess_sentence_no_stem)
    df['nostem_2'] = df['s2'].apply(preprocess_sentence_no_stem)
    df['token_m'] = df['sm'].apply(preprocess_sentence)  # vecteur
    df['s1_pp'] = df['token_1'].apply(' '.join)  # string
    df['s2_pp'] = df['token_2'].apply(' '.join)  # string
    df['sm_pp'] = df['token_m'].apply(' '.join)  # string
    df['GroundTruthScore'] = df['scores'] / 5  # normalisation du score entre 0 et 1
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

    df.to_pickle("preproc.pkl")

    return df

    
def save_numpy_array(array, filename):
    
    #save numpy array
    np.save(filename, array)
    
def load_numpy_array(filename):
    if filename is None:
        raise ValueError("filename must be a non empty string")
    
    #load numpy array
    return np.load(filename)


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
    for i in df.index:
        s1 = df["s1"][i]
        s2 = df["s2"][i]
        if (len(s1)) == 0 or (len(s2)==0):
            jacc_score = 0
        else:
            # binary=True because we use Jaccard score (we want presence/absence information, not counts)
            cv = CountVectorizer(binary=True)
            vectors = cv.fit_transform([s1, s2]).toarray()
            jacc_score = jaccard_score(vectors[0], vectors[1])

        df.loc[i, 'scores_0'] = jacc_score

    return df

from gensim.models import KeyedVectors


def feature_rougeL(df):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) #, 'rougeS'

    for i in df.index:
        reference_summary = df['s1_pp'][i]
        candidate_summary = df['s2_pp'][i]
        scores = scorer.score(str(reference_summary), str(candidate_summary))

        df.loc[i, 'scores_rougeL'] = scores['rougeL'].fmeasure

    return df

def WMD_score(df,m=1):
    """Calcule la distance WMD et les scores associé entre les phrases de 2 colonnes
    Attributes:
        df: The pandas dataframe
        m: coefficient for the exponential score
    """
    

#path = 'c:\Users\anaele.baudant\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin\'
    file='GoogleNews-vectors-negative300.bin'

    # Load vectors directly from the file
    model = KeyedVectors.load_word2vec_format(file, binary=True) 
    for i, (s1v,s2v) in enumerate(zip(df["nostem_1"],df["nostem_2"])):

        #retire les mots pas dans le vocabulaire
        s1v=[word for word in s1v if word in model.key_to_index]
        s2v=[word for word in s2v if word in model.key_to_index]
        #calcule distance WMD
        distance_wmd = model.wmdistance(s1v,s2v)
        df.loc[i, 'distance_WMD'] = distance_wmd / ((len(s1v)*len(s2v))+1)
        #transformation distance en score normalisé (entre 0 et 1)
        df.loc[i, 'scores_WMD_inv'] = 1-distance_wmd
        #df.loc[i, 'scores_WMD_exp'] = np.exp(-distance_wmd*m)
    #df['scores_WMD_max'] = 1-df['distance_WMD']/df['distance_WMD'].max()
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
        #df.loc[i, 'scores_1_eucl'] = d_eucl
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
    vectorizers = [CountVectorizer(ngram_range=(1, 3))]

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
            #df[f'scores_5_euclidean_{i+1}_syn'] = euclidians
            df[f'scores_5_cosine_{i+1}_syn'] = cosines
        else:
            #df[f'scores_4_euclidean_{i+1}'] = euclidians
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
#ANAELE-from co_occurrence_matrix import create_vocabulary, co_occurence_matrix, assign_distributional_vectors
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
    preprocessed_dataset = WMD_score(preprocessed_dataset,m=25)
    preprocessed_dataset = feature_rougeL(preprocessed_dataset)
    return preprocessed_dataset


def post_process_data(df):
    # select columns that need post-processing: those that start with 'scores_'
    # Ahmed here : I suggest to just drop the columns that we don't need by name
    # because i have many variants of scores and i can't align to this naming convention
    columns_to_normalize = []

    logger.info("There are %s columns", df.columns)
    for col in df.columns:
        if col.startswith('scores_'):
            columns_to_normalize.append(col)

    if(len(columns_to_normalize) == 0):
        raise ValueError("No columns to normalize")
    
    logger.info("Normalizing columns: %s", columns_to_normalize)
    df[columns_to_normalize] = StandardScaler().fit_transform(df[columns_to_normalize])
    return df


def shared_memory_array(data):
    # Create shared memory and copy data into it
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    np.copyto(shared_data, data)
    return shm, shared_data

def test_model_shared(model, shm_names, combo_indices, shape_train, shape_test, shape_ytrain, shape_ytest):
    # Access shared memory
    X_train_shm = shared_memory.SharedMemory(name=shm_names['X_train'])
    X_test_shm = shared_memory.SharedMemory(name=shm_names['X_test'])
    y_train_shm = shared_memory.SharedMemory(name=shm_names['y_train'])
    y_test_shm = shared_memory.SharedMemory(name=shm_names['y_test'])

    # Reconstruct arrays from shared memory
    # Assuming data types and shapes are known, replace with actual types and shapes
    X_train = np.ndarray(shape_train, dtype=np.float64, buffer=X_train_shm.buf)[:, combo_indices]
    X_test = np.ndarray(shape_test, dtype=np.float64, buffer=X_test_shm.buf)[:, combo_indices]
    y_train = np.ndarray(shape_ytrain, dtype=np.float64, buffer=y_train_shm.buf)
    y_test = np.ndarray(shape_ytest, dtype=np.float64, buffer=y_test_shm.buf)

    # Test the model (replace with actual testing logic)
    model = model()
    result = test_model(combo_indices, model, X_train, X_test, y_train, y_test)

    # Close shared memory (data remains accessible)
    X_train_shm.close()
    X_test_shm.close()
    y_train_shm.close()
    y_test_shm.close()

    return result


def test_model(feature_combination, model, X_train, X_test, y_train, y_test):

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)
    prediction_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, prediction_train,multioutput='raw_values')
    r2_train = r2_score(y_train, prediction_train)
    mse_test = mean_squared_error(y_test, predictions, multioutput='raw_values')
    r2_test = r2_score(y_test, predictions)
    results = {
             'features': feature_combination,
             'model': model.__class__.__name__,
             'MSE_train': mse_train,
             'MSE': mse_test,
             'R2_train': r2_train,
             'R2': r2_test
         }
    
    if hasattr(model, 'coef_'):
        results['thetas'] = model.coef_

    if model.__class__.__name__ == ("LassoCV" or "ElasticNetCV" or "RidgeCV"):
        if hasattr(model, 'alpha_'):
            results['alpha'] = model.alpha_

    return results

def all_feature_combinations(train_features):
    """
    Generate all possible combinations of feature indices from the given design matrix.

    Parameters:
    train_features (array-like): The design matrix with N features.

    Returns:
    list of tuples: A list containing tuples, each tuple represents a combination of feature indices.
    """
    n_features = train_features.shape[1]
    feature_indices = range(n_features)
    all_combinations = []

    # Generate combinations for all possible lengths
    for r in range(1, n_features + 1):
        all_combinations.extend(combinations(feature_indices, r))

    return all_combinations



def run_tests_parallel(models_to_test, X_train, X_test, y_train, y_test):
    feature_combos = all_feature_combinations(X_train)
    results = []

    logger.info("Creating shared memory")
    X_train_shm, X_train_shared = shared_memory_array(X_train)
    X_test_shm, X_test_shared = shared_memory_array(X_test)
    y_train_shm, y_train_shared = shared_memory_array(y_train)
    y_test_shm, y_test_shared = shared_memory_array(y_test)

    shm_names = {
        'X_train': X_train_shm.name,
        'X_test': X_test_shm.name,
        'y_train': y_train_shm.name,
        'y_test': y_test_shm.name
    }
    logger.info("There are %s feature combinations to test", len(feature_combos))
    for model in models_to_test:
        logger.info("Testing model: %s", model)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(test_model_shared, model, shm_names, combo, X_train.shape, X_test.shape, y_train.shape, y_test.shape) 
                       for combo in feature_combos]
            
            for future in futures:
                results.append(future.result())
    
    # Cleanup shared memory
    X_train_shm.close()
    X_train_shm.unlink()
    X_test_shm.close()
    X_test_shm.unlink()
    y_train_shm.close()
    y_train_shm.unlink()
    y_test_shm.close()
    y_test_shm.unlink()
    return pd.DataFrame(results)

def is_fitted(estimator):
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False
    
def run_other_tests_in_parallel(X_train, X_test, y_train, y_test):
    
    lassoCV_r2 = LassoCV(alphas=np.linspace(0.0001, 1000, 100), cv=10, max_iter=10000)
    elasticNetCV_r2 = ElasticNetCV(alphas=np.linspace(0.0001, 1000, 100), cv=10, max_iter=10000)
    ridgeCV_r2 = RidgeCV(alphas=np.linspace(0.0001, 1000, 100), cv=10)
    models = [lassoCV_r2, elasticNetCV_r2, ridgeCV_r2]

    results = []
    logger.info("Testing other models")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(test_model, [], model, X_train, X_test, y_train, y_test) for model in models]
        
        for future in futures:
            results.append(future.result())
    

    return pd.DataFrame(results)

def test_models_simple(dataset, preprocess_function, postprocess_function):
    if "postprocessed_train.pkl" in os.listdir() and "train_features.npy" in os.listdir() and "test_features.npy" in os.listdir() and "train_features.npy" in os.listdir() and "test_features.npy" in os.listdir():
        logger.info("Loading train_features.npy")
        train_features = load_numpy_array("train_features.npy")

        logger.info("Loading test_features.npy")
        test_features = load_numpy_array("test_features.npy")

        logger.info("Loading score_to_predict.npy")
        score_to_predict = load_numpy_array("score_to_predict.npy")

        logger.info("Loading score_to_verify.npy")
        score_to_verify = load_numpy_array("score_to_verify.npy")

        logger.info("postprocessed_train.pkl")
        postprocessed_train = pd.read_pickle("postprocessed_train.pkl")

    else:

        logger.info("preprocessing training set")
        preprocessed_train = preprocess_function(dataset['train'])


        logger.info("postprocessing training set")
        postprocessed_train = postprocess_function(preprocessed_train)

        logger.info("preprocessing test set")
        preprocessed_test = preprocess_function(dataset['test'])

        logger.info("postprocessing test set")
        postprocessed_test = postprocess_function(preprocessed_test)

        # Fill NaN values if any
        postprocessed_train = postprocessed_train.fillna(0)
        postprocessed_test = postprocessed_test.fillna(0)

        # Extract features and targets
        logger.info("Extracting features and targets")
        # log scores name
        logger.info("Scores names: %s", [col for col in postprocessed_train.columns if col.startswith('scores_')])

        train_features = postprocessed_train[[col for col in postprocessed_train.columns if col.startswith('scores_')]].to_numpy()

        test_features = postprocessed_test[[
            col for col in postprocessed_test.columns if col.startswith('scores_')]].to_numpy()
        score_to_predict = postprocessed_train['GroundTruthScore']
        score_to_verify = postprocessed_test['GroundTruthScore']

        postprocessed_train.to_pickle("postprocessed_train.pkl")
        save_numpy_array(train_features, "train_features.npy")
        save_numpy_array(test_features, "test_features.npy")
        save_numpy_array(score_to_predict, "score_to_predict.npy")
        save_numpy_array(score_to_verify, "score_to_verify.npy")

    # Iterate over all combinations of features

    scores_name = dict()
    i = 0
    for col, indice in zip(postprocessed_train.columns, range(len(postprocessed_train.columns))):
        if col.startswith('scores_'):
            scores_name[i] = col
            i+=1
    logger.debug("Running tests in parallel")

    #inspect the data here
    df = pd.DataFrame(train_features)
    df2 = pd.DataFrame(test_features)

    #plot score_0
    plt.plot(df.iloc[:,0], label='train')
    plt.show()
    plt.plot(df2.iloc[:,0], label='test')
    plt.show()
    # run linear regression for each feature
    model = LinearRegression()
    results = []
    for i in range(train_features.shape[1]):
        model.fit(sm.add_constant(train_features[:,i]), score_to_predict)
        prediction_train = model.predict(sm.add_constant(train_features[:,i]))
        prediction = model.predict(sm.add_constant(test_features[:,i]))

        mse = mean_squared_error(score_to_verify, prediction)
        r2 = r2_score(score_to_verify, prediction)
        result = {
            'features': [i],
            'model': model.__class__.__name__,
            'MSE_train': mean_squared_error(score_to_predict,prediction_train),
            'MSE': mean_squared_error(score_to_verify, prediction),
            'R2_train': r2_score(score_to_predict, prediction_train),
            'R2': r2_score(score_to_verify, prediction),
            'thetas': model.coef_
        }
        results.append(result)
    single_feature_results = pd.DataFrame(results)

    # run ridge lasso and elastic net for the whole dataset
    linear = LinearRegression()
    lassoCV_r2 = LassoCV(alphas=np.log(np.logspace(0.001, 100, 100)), cv=10, max_iter=10000)
    elasticNetCV_r2 = ElasticNetCV(alphas=np.log(np.logspace(0.001, 100, 100)), cv=10, max_iter=10000)
    ridgeCV_r2 = RidgeCV(alphas=np.log(np.logspace(0.001, 100, 100)), cv=10)
    models = [linear, lassoCV_r2, elasticNetCV_r2, ridgeCV_r2]

    results = []
    logger.info("Testing other models")
    for model in models:
        model.fit(train_features, score_to_predict)
        prediction_train = model.predict(train_features)
        prediction = model.predict(test_features)
        mse = mean_squared_error(score_to_verify, prediction)
        r2 = r2_score(score_to_verify, prediction)
        result = {
            'features': [i for i in range(train_features.shape[1])],
            'model': model.__class__.__name__,
            'MSE_train': mean_squared_error(score_to_predict,prediction_train),
            'MSE': mean_squared_error(score_to_verify, prediction),
            'R2_train': r2_score(score_to_predict, prediction_train),
            'R2': r2_score(score_to_verify, prediction),
            'thetas': model.coef_
        }
        if model.__class__.__name__ == ("LassoCV" or "ElasticNetCV" or "RidgeCV"):
            if hasattr(model, 'alpha_'):
                result['alpha'] = model.alpha_
        results.append(result)
    
    whole_dataset_results = pd.DataFrame(results)
    return single_feature_results, whole_dataset_results, scores_name


def test_models_with_feature_combinations(dataset, preprocess_function, postprocess_function):

    if "postprocessed_train.pkl" in os.listdir() and "train_features.npy" in os.listdir() and "test_features.npy" in os.listdir() and "train_features.npy" in os.listdir() and "test_features.npy" in os.listdir():
        logger.info("Loading train_features.npy")
        80

        logger.info("Loading test_features.npy")
        test_features = load_numpy_array("test_features.npy")

        logger.info("Loading score_to_predict.npy")
        score_to_predict = load_numpy_array("score_to_predict.npy")

        logger.info("Loading score_to_verify.npy")
        score_to_verify = load_numpy_array("score_to_verify.npy")

        logger.info("postprocessed_train.pkl")
        postprocessed_train = pd.read_pickle("postprocessed_train.pkl")

    else:

        logger.info("preprocessing training set")
        preprocessed_train = preprocess_function(dataset['train'])


        logger.info("postprocessing training set")
        postprocessed_train = postprocess_function(preprocessed_train)

        logger.info("preprocessing test set")
        preprocessed_test = preprocess_function(dataset['test'])

        logger.info("postprocessing test set")
        postprocessed_test = postprocess_function(preprocessed_test)

        # Fill NaN values if any
        postprocessed_train = postprocessed_train.fillna(0)
        postprocessed_test = postprocessed_test.fillna(0)

        # Extract features and targets
        logger.info("Extracting features and targets")
        # log scores name
        logger.info("Scores names: %s", [col for col in postprocessed_train.columns if col.startswith('scores_')])

        train_features = postprocessed_train[[col for col in postprocessed_train.columns if col.startswith('scores_')]].to_numpy()

        test_features = postprocessed_test[[
            col for col in postprocessed_test.columns if col.startswith('scores_')]].to_numpy()
        score_to_predict = postprocessed_train['GroundTruthScore']
        score_to_verify = postprocessed_test['GroundTruthScore']

        postprocessed_train.to_pickle("postprocessed_train.pkl")
        save_numpy_array(train_features, "train_features.npy")
        save_numpy_array(test_features, "test_features.npy")
        save_numpy_array(score_to_predict, "score_to_predict.npy")
        save_numpy_array(score_to_verify, "score_to_verify.npy")

    # Iterate over all combinations of features

    scores_name = dict()
    i = 0
    for col, indice in zip(postprocessed_train.columns, range(len(postprocessed_train.columns))):
        if col.startswith('scores_'):
            scores_name[i] = col
            i+=1
    logger.debug("Running tests in parallel")
    models_to_test_combinations = [LinearRegression, RidgeCV, DecisionTreeRegressor, LassoCV, ElasticNetCV]
    if("result1.pkl" in os.listdir()):
        logger.info("Loading result1.pkl")
        result1 = pd.read_pickle("result1.pkl")
    else:
        result1 = run_tests_parallel(models_to_test_combinations, train_features, test_features, score_to_predict, score_to_verify)
        pd.DataFrame(result1).to_pickle("result1.pkl")

    
    return result1, scores_name
