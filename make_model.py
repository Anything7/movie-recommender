import ast
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies = movies.dropna()

def convert(text):
    return [i["name"] for i in ast.literal_eval(text)]


def convert_cast(text):
    return [i["name"] for i in ast.literal_eval(text)][:3]


def fetch_director(text):
    return [i["name"] for i in ast.literal_eval(text) if i["job"] == "Director"]


def collapse(names):
    return [n.replace(" ", "") for n in names]


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(fetch_director)

for col in ["cast", "crew", "genres", "keywords"]:
    movies[col] = movies[col].apply(collapse)

movies["overview"] = movies["overview"].apply(lambda x: x.split())

movies["tags"] = (
    movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
)
new = movies.drop(columns=["overview", "genres", "keywords", "cast", "crew"]).reset_index(drop=True)
new["tags"] = new["tags"].apply(lambda x: " ".join(x).lower())

cv = CountVectorizer(max_features=5000, stop_words="english")
vector = cv.fit_transform(new["tags"]).toarray()
similarity = cosine_similarity(vector)

os.makedirs("model", exist_ok=True)
pickle.dump(new, open("model/movie_list.pkl", "wb"))
pickle.dump(similarity, open("model/similarity.pkl", "wb"))
print("Done! Saved model/movie_list.pkl and model/similarity.pkl")
print("Movies:", len(new))