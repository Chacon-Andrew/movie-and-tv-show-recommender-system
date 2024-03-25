import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity


# def explore_data():
#     df = pd.read_csv("./Data/tmdb_all_data.csv", lineterminator='\n')
#     new_df = df.dropna()
#     for d in new_df:
#         print(d)
#     print(new_df['video'])

def train():
    #read in data
    df = pd.read_csv("./Data/tmdb_all_data.csv", lineterminator='\n')

    #combine relevant data into one column
    df['content'] = df['title'].astype(str) + ' ' + df['overview'].astype(str) + df['genres'] + df['vote_average'].astype(str) + df['vote_count'].astype(str) + df['production_companies'].astype(str) + df['budget'].astype(str)
    df['content'] = df['content'].fillna('')

    #tokenize content
    df['tokenized_content'] = df['content'].apply(simple_preprocess)
    #initialize word2vec model
    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)

    #build vocabulary
    model.build_vocab(df['tokenized_content'])

    #train
    model.train(df['tokenized_content'], total_examples=model.corpus_count, epochs=50)

    #save the model
    model.save('./Data/recommendation.model')

def load_model():
    model = Word2Vec.load('./Data/recommendation.model')
    return model

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features), dtype="float64")
    nwords = 0

    for word in words:
        if word in vocabulary:
            nwords += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector

def average_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)

def reccommend(model, movie):
    #read in data
    df = pd.read_csv("./Data/tmdb_all_data.csv", lineterminator='\n')

    #combine relevant data into one column
    df['content'] = df['title'].astype(str) + ' ' + df['overview'].astype(str) + ' ' + df['genres'] + ' ' + df['vote_average'].astype(str) + ' ' + df['vote_count'].astype(str) + ' ' + df['production_companies'].astype(str) + ' ' + df['budget'].astype(str)
    df['content'] = df['content'].fillna('')
    #tokenize content
    df['tokenized_content'] = df['content'].apply(simple_preprocess)
    w2v_feature_array = average_word_vectorizer(corpus=df['tokenized_content'], model=model, num_features=100)
    movie_index = df[df['title'] == movie].index[0]
    user_movie_vector = w2v_feature_array[movie_index].reshape(1, -1)
    similarity_scores = cosine_similarity(user_movie_vector, w2v_feature_array)

    similar_movies = list(enumerate(similarity_scores[0]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:20]
    return sorted_similar_movies

if __name__ == '__main__':
    # train()
    model = load_model()
    recommened_list = reccommend(model, 'The Dark')
    df = pd.read_csv("./Data/tmdb_all_data.csv", lineterminator='\n')
    for i, score in recommened_list:
        print("{}: {}".format(i, df.loc[i, 'title']))