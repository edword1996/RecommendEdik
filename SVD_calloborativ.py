import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
from scipy.sparse.linalg import svds

header = ['userid', 'moveid', 'rating', 'timestamp']
df = pd.read_csv('ratings.csv', sep='\t', names=header)
print(df.head())

#считаем количество уникальных пользователей и фильмов:
n_users = df.userid.unique().shape[0]
n_items = df.moveid.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

# разбиваем выборку на трейн и тест, ставим тестовый процент 0.25
train_data, test_data = train_test_split(df, test_size=0.25)

# создаем две матрицы элементов пользователя, одну для обучения и другую для тестирования
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
#для тестирования
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#использую функцию pairwise_distances  для вычисления сходства косинусов.
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

#делаем прогнозы. у нас есть матрицы подобия: user_simility и item_simility, и поэтому
# делаем прогноз, применив  формулу для CF на основе пользователя:
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)

        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

#учитываю только прогнозируемые рейтинги, которые есть в наборе тестовых данных,
# отфильтровал все остальные элементы в матрице прогнозирования с  [ground_truth.nonzero ()].
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

# SVD
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print('The sparsity level of Movie is ' +  str(sparsity*100) + '%')

u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print ('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))

