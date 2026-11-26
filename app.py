import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st

API_KEY = "8eb8afb96cff6762b955b91be75c7b19"


@st.cache_resource
def get_poster_cache():
    return {}


def fetch_poster(movie_id):
    cache = get_poster_cache()
    movie_id = int(movie_id)
    if movie_id in cache:
        return cache[movie_id] or None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    for attempt in range(4):  # try up to 4 times
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                poster_path = response.json().get("poster_path")
                full_path = ("https://image.tmdb.org/t/p/w500" + poster_path) if poster_path else ""
                cache[movie_id] = full_path  # remember it
                return full_path or None
            time.sleep(1 + attempt)  # server busy, wait and retry
        except Exception:
            time.sleep(1 + attempt)  # network problem, wait and retry
    return None  # failed 4 times (not remembered, so it retries next time)


def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    top = distances[1:6]
    ids = [movies.iloc[i[0]].movie_id for i in top]
    names = [movies.iloc[i[0]].title for i in top]
    with ThreadPoolExecutor(max_workers=5) as pool:  # fetch all 5 posters together
        posters = list(pool.map(fetch_poster, ids))
    return names, posters


st.header("Movie Recommender System")

movies = pickle.load(open("model/movie_list.pkl", "rb"))
similarity = pickle.load(open("model/similarity.pkl", "rb"))

selected_movie = st.selectbox("Type or select a movie from the dropdown", movies["title"].values)

if st.button("Show Recommendation"):
    names, posters = recommend(selected_movie)
    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.text(name)
            if poster:
                st.image(poster)
            else:
                st.caption("No poster available")