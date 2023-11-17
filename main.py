from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
from features import load_data, preprocess_sentence, preprocess_dataset, get_corpus, get_words, get_vocabulary, feature0_scores, feature1_scores, feature2_scores, feature3_scores, n_gram_overlap, feature6_scores, feature7_scores, show_scores, show_top_error, show_examples 
from features import extract_features, post_process_data, test_models_with_feature_combinations

from logger import logger
import matplotlib.pyplot as plt


def test_something():
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
    preprocessed_train = feature0_scores(preprocessed_train)
    preprocessed_train = feature1_scores(preprocessed_train)
    preprocessed_train = feature2_scores(preprocessed_train)
    preprocessed_train = feature3_scores(preprocessed_train)
    # feature 4 = n_gram_overlap
    preprocessed_train = n_gram_overlap(preprocessed_train, corpus)
    preprocessed_train = feature6_scores(preprocessed_train)
    preprocessed_train = feature7_scores(preprocessed_train)

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

def run_models():
    logger.info(f"Starting to load data")
    dataset = load_data()

    another_set = {
        'train': dataset['train'],
        'test': dataset['test']
    }

    logger.info(f"Starting to test models")
    results_df = test_models_with_feature_combinations(another_set, extract_features, post_process_data)

    #save results in python
    results_df.to_pickle("results.pkl")



# logger.debug(f"Starting to combine model. Models: %s", models_to_test)
try:
    #test_something()
    run_models()

    # load results from python
    result = pd.read_pickle("results.pkl")

    top_10_df = result.nlargest(10, 'R2')

    print(top_10_df)

except Exception as e:
    print(e)
    logger.error(f"Error in main.py: %s", e)
    raise e



