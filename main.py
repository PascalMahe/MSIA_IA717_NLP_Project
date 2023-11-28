from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
from features import load_data, preprocess_sentence, preprocess_dataset, get_corpus, get_words, get_vocabulary, feature0_scores, feature1_scores, feature2_scores, feature3_scores, n_gram_overlap, feature6_scores, feature7_scores, show_scores, show_top_error, show_examples 
from features import test_models_simple, extract_features, post_process_data

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
    #results_df , score_names = test_models_with_feature_combinations(another_set, extract_features, post_process_data)
    result_single_feature, result_multi , score_names = test_models_simple(another_set,extract_features, post_process_data)
    #save results in python
    result_single_feature.to_pickle("result_single.pkl")
    result_multi.to_pickle("results_multi.pkl")
    save_score_names = pd.DataFrame(list(score_names.values()), index=score_names.keys(), columns=['Score Names'])
    save_score_names.to_pickle("score_names.pkl")

def get_top_coef(top_25):
    # get the first top R2 score from each model
    result = top_25.sort_values(by=['R2'], ascending=False)
    result = result.drop_duplicates(subset=['model'], keep='first')
    return result

def get_top_single_feature_scores(df):
    # select only df rows with one feature
    df = df[df['features'].map(len) == 1]
    # sort by R2
    df = df.sort_values(by=['R2'], ascending=False)
    #get list of models
    models = df['model'].unique()
    #get list of features
    features = df['features'].unique()
    #for each models and each features, get the best R2 score
    result = pd.DataFrame(columns=df.columns)
    for model in models:
        for feature in features:
            r = df[(df['model'] == model) & (df['features'] == feature)]
            r = r.sort_values(by=['R2'], ascending=False)
            r = r.head(1)
            result = pd.concat([result, r], ignore_index=True)
    return result

import pandas as pd
import matplotlib.pyplot as plt

def plot_thetas_from_df(df):
    # Check if DataFrame has the required columns
    required_columns = {'features', 'model', 'MSE', 'R2', 'thetas'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Determine number of rows in DataFrame
    num_rows = len(df)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(10, num_rows * 4))
    
    # If only one row, wrap axes in a list
    if num_rows == 1:
        axes = [axes]

    colors = ['lightblue', 'lightgreen', 'pink', 'lightyellow', 'lightcyan', 'lightgray', 'lightcoral', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow']
    for i, row in enumerate(df.itertuples()):
        ax = axes[i]
        if (len(row.features) +1) == len(row.thetas):
            # Add a row for the intercept
            row.features.insert(0, 'Intercept')
        label = f'Model: {row.model} - MSE: {row.MSE:.5f}, R2: {row.R2:.5f}'
        bars = ax.bar(row.features, row.thetas, label=label, color=colors[i])
        ax.set_ylabel('Theta Values')

        # Annotate each bar with its value
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), 
                    verticalalignment='bottom', ha='center', fontsize=8)

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=2)

    # Add legend to top right of plot
    fig.subplots_adjust(right=0.8)

    # Create a common legend in the top right corner outside the subplots
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, fontsize=8)
    plt.tight_layout()
    plt.show()

def replace_indice_with_name(df, score_names):
    for index, row in df.iterrows():
        features = row['features']
        logger.info('features: %s', features)

        my_new_tuple = []
        for indice in features:
            logger.info('indice: %s', indice)
            logger.info('score_names: %s', score_names[indice])
            my_new_tuple.append(score_names[indice])

        df.at[index, 'features'] = my_new_tuple

    return df
from pandas.plotting import table
def save_df_as_plot(df, name):
    fig, ax = plt.subplots(figsize=(5, 3))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    tbl = table(ax, df, loc='center')

    # Increase font size
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)  # Set to your desired font size
    tbl.scale(1.2, 1.2)  # You can also scale the table elements like cell size

    plt.suptitle(name, fontsize=16)  # Set your title and font size here

    plt.show()

def post_processing_parallel():

    ## load results from python
    result = pd.read_pickle("results.pkl")
    score_names = pd.read_pickle("score_names.pkl")
    score_names = score_names.to_dict(orient='index')
    score_names = {k: v['Score Names'] for k, v in score_names.items()}
    print(score_names)
    top_coef = get_top_coef(result[result['model'] != 'DecisionTreeRegressor'])
    single_feature = get_top_single_feature_scores(result[result['model'] != 'DecisionTreeRegressor'])
    #replace indice id by column name
    top_coef = replace_indice_with_name(top_coef, score_names)

    single_feature = replace_indice_with_name(single_feature, score_names)
    print("____________Top single feature_____________")
    print(single_feature[['model','features','R2_train','R2','MSE_train', 'MSE']])
    models = single_feature['model'].unique()
    for model in models:
        df = single_feature[single_feature['model'] == model]
        save_df_as_plot(df[['features','R2_train','R2','MSE_train', 'MSE']], model)
    print("____________Top coef_____________")
    print(top_coef[['model','R2_train','R2','MSE_train', 'MSE']])
    save_df_as_plot(top_coef[['model', 'R2_train','R2','MSE_train', 'MSE']], 'top_coef')
    # display in a table R2 and MSE for each model 


# logger.debug(f"Starting to combine model. Models: %s", models_to_test)
try:
    #test_something()
    run_models()

    single_features = pd.read_pickle("result_single.pkl")
    multi_features = pd.read_pickle("results_multi.pkl")
        
    score_names = pd.read_pickle("score_names.pkl")
    score_names = score_names.to_dict(orient='index')
    score_names = {k: v['Score Names'] for k, v in score_names.items()}

    single_features = replace_indice_with_name(single_features, score_names)
    multi_features = replace_indice_with_name(multi_features, score_names)
    print("____________Top single feature_____________")
    save_df_as_plot(single_features[['features','R2_train','R2','MSE_train', 'MSE']], "LinearRegression")

    print("____________Top coef_____________")
    print(multi_features[['model','R2_train','R2','MSE_train', 'MSE']])
    save_df_as_plot(multi_features[['model', 'R2_train','R2','MSE_train', 'MSE']], 'top_coef')

    plot_thetas_from_df(multi_features)


except Exception as e:
    print(e)
    logger.error(f"Error in main.py: %s", e)
    raise e



