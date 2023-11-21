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
    results_df , score_names = test_models_with_feature_combinations(another_set, extract_features, post_process_data)

    #save results in python
    results_df.to_pickle("results.pkl")
    save_score_names = pd.DataFrame(list(score_names.values()), index=score_names.keys(), columns=['Score Names'])
    save_score_names.to_pickle("score_names.pkl")

def get_top_coef(top_25):
    # get the first top R2 score from each model
    result = top_25.sort_values(by=['R2'], ascending=False)
    result = result.drop_duplicates(subset=['model'], keep='first')
    return result


import matplotlib.pyplot as plt

def plot_thetas_from_df(df):
    # Check if DataFrame has the required columns
    required_columns = {'features', 'model', 'MSE', 'R2', 'thetas'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Plotting
    fig, axes = plt.subplots(nrows=len(df), ncols=1, figsize=(10, len(df) * 4))

    for i, row in df.iterrows():
        ax = axes[i] if len(df) > 1 else axes
        ax.bar(row['features'], row['thetas'])
        ax.set_title(f'Model: {row["model"]} - MSE: {row["MSE"]:.5f}, R2: {row["R2"]:.5f}')
        ax.set_ylabel('Theta Values')
        ax.set_xlabel('Features')

    plt.tight_layout()
    plt.show()

# Example usage with a DataFrame 'df'
# df = pd.read_your_dataframe_here()  # Replace with your DataFrame loading method
# plot_thetas_from_df(df)


# logger.debug(f"Starting to combine model. Models: %s", models_to_test)
try:
    #test_something()
    #run_models()

    ## load results from python
    result = pd.read_pickle("results.pkl")
    score_names = pd.read_pickle("score_names.pkl")
    score_names = score_names.to_dict(orient='index')
    score_names = {k: v['Score Names'] for k, v in score_names.items()}
    print(score_names)
    top_25_df = result[result['model'] != 'DecisionTreeRegressor']
    top_25_df = top_25_df.nlargest(25, 'R2')

    #replace indice id by column name
    for index, row in top_25_df.iterrows():
        features = row['features']
        logger.info('features: %s', features)

        my_new_tuple = []
        for indice in features:
            logger.info('indice: %s', indice)
            logger.info('score_names: %s', score_names[indice])
            my_new_tuple.append(score_names[indice])

        top_25_df.at[index, 'features'] = my_new_tuple
    
    print(my_new_tuple)
    top_25_df.at[index, 'features'] = my_new_tuple


    print(top_25_df)

    #print scores mse model and features
    print(top_25_df[['model', 'features', 'R2', 'MSE']])

    r = get_top_coef(top_25_df)
    print(r[['model', 'features', 'R2', 'thetas']])

    plot_thetas_from_df(r)

except Exception as e:
    print(e)
    logger.error(f"Error in main.py: %s", e)
    raise e



