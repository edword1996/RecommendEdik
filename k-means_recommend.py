from sklearn import linear_model
import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score

n_features = 10000 # Максимальное количество слов для использования
min_df = 2 # Минимальная частота документов
max_df = 0.8 # Максимальная доля документов, в которых может встречаться слово
n_clusters = 10 #Количество кластеров, чтобы найти


df = pd.read_csv('tmdb_5000_movies.csv')
df.dropna(subset = ['overview'],inplace=True)
print("Length of Dataset ",len(df))
print(df.head(2))



# Конвертировать обзоры фильмов из текста в векторы TF-IDF
vectorizer = TfidfVectorizer(max_df=max_df,
                             max_features=n_features,
                             min_df=min_df,
                             stop_words='english',
                             use_idf=True)
X = vectorizer.fit_transform(df.overview.values)
print(X.shape)

km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                verbose=False)
print(km.fit(X))


df['cluster_id'] = km.predict(X)
clusters = df.groupby('cluster_id')['original_title']
def get_movies(cluster_id,n_movies):
  movies=[]
  for movie in clusters.get_group(cluster_id)[0:n_movies]:
    movies.append(movie)
  return movies


print("Movie Clusters\n\n")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(n_clusters):
    print("Cluster %d" % i)
    print("Terms:" , end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print("\nMovies :" , get_movies(i,5))
    print()






