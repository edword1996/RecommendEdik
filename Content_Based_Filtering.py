import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from sklearn.metrics import mean_squared_error
from math import sqrt

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

df1.columns = ['id','tittle','cast','crew']
df= df2.merge(df1,on='id')


tfidf = TfidfVectorizer(stop_words = 'english')

# Меняю NaN пустой строкой
df['overview'] = df['overview'].fillna('')

# Генерация матрицы TF-IDF путем подгонки и преобразования данных
tfidf_matrix = tfidf.fit_transform(df['overview'])


print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#print(cosine_sim)


indices = pd.Series(df.index, index = df['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    # получить индекс фильма, который соответствует названию фильма
    idx = indices[title]

    # получить попарные оценки сходства всех фильмов с интересующим фильмом
    sim_scores = list(enumerate(cosine_sim[idx]))

    # сортировка фильмов по показателям сходства
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # получить оценки из 10 самых похожих фильмов
    sim_scores = sim_scores[1:11]

    # получить индексы фильмов
    movie_indices = [i[0] for i in sim_scores]

    # верните топ 10 самых похожих фильмов
    return df['title'].iloc[movie_indices]

print(get_recommendations('Forrest Gump'))









