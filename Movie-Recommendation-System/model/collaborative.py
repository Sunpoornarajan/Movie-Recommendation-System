import pandas as pd
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv("data/ratings.csv")

# User-Movie Matrix
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# KNN Model
knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(user_movie_matrix)

def recommend_movies(user_id, k=2):
    if user_id not in user_movie_matrix.index:
        return []

    user_index = user_movie_matrix.index.get_loc(user_id)

    distances, indices = knn.kneighbors(
        [user_movie_matrix.iloc[user_index]],
        n_neighbors=k + 1
    )

    similar_users = indices.flatten()[1:]
    recommendations = set()

    for user in similar_users:
        liked_movies = user_movie_matrix.iloc[user]
        liked_movies = liked_movies[liked_movies >= 4].index
        recommendations.update(liked_movies)

    return list(recommendations)