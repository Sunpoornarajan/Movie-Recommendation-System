from flask import Flask, render_template, request
import pandas as pd
import webbrowser
import threading

from model.collaborative import recommend_movies
from model.similarity import recommend_popular

app = Flask(__name__)

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Average rating per movie
avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()

def get_movie_details(movie_ids):
    data = movies[movies["movieId"].isin(movie_ids)]
    data = data.merge(avg_ratings, on="movieId", how="left")
    data["rating"] = data["rating"].round(1)
    return data.to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def home():
    movie_list = []

    if request.method == "POST":
        user_id = request.form["user_id"]

        if user_id.strip() == "":
            movie_ids = recommend_popular(6)
        else:
            movie_ids = recommend_movies(int(user_id), k=3)
            if not movie_ids:
                movie_ids = recommend_popular(6)

        movie_list = get_movie_details(movie_ids)

    return render_template("index.html", movies=movie_list)

# 🔥 AUTO OPEN BROWSER
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)