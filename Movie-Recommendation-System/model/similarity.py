import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# Popular Movies (Cold Start)
popular_movies = ratings.groupby("movieId")["rating"].mean().sort_values(ascending=False)

def recommend_popular(top_n=3):
    return popular_movies.head(top_n).index.tolist()

# Genre-Based Similarity
tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movies["genres"])
genre_similarity = cosine_similarity(genre_matrix)

def genre_recommend(movie_id, top_n=3):
    idx = movies[movies["movieId"] == movie_id].index[0]
    similarity_scores = list(enumerate(genre_similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return movies.iloc[movie_indices]["movieId"].tolist()