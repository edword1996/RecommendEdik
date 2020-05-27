import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import mean_squared_error


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)


df = pd.read_csv('tmdb_5000_credits.csv')
df1 = pd.read_csv('tmdb_5000_movies.csv')

df.columns = ['id','tittle','cast','crew']
df1= df1.merge(df,on='id')
#print(df1.head())

C= df1['vote_average'].mean()
#print(C)


C= df1['vote_average'].mean()
print('C = ',C)
m= df1['vote_count'].quantile(0.9)
print('m = ',m)

q_movies = df1.copy().loc[df1['vote_count'] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))








