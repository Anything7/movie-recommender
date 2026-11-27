import ast            #Converts string data into Python objects.
import os             #	Creates folders and handles file paths.
import pickle         #Saves the trained data for future use.

import pandas as pd            # for reading dataset
from sklearn.feature_extraction.text import CountVectorizer        #Converts text into numerical vectors
from sklearn.metrics.pairwise import cosine_similarity            #Finds similarities between movies.

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies = movies.dropna()

def convert(text):                #Creates a function called convert.
    return [i["name"] for i in ast.literal_eval(text)]                #Converts string data into a Python list.


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
    movies[col] = movies[col].apply(collapse)     #This removes spaces from names. 

movies["overview"] = movies["overview"].apply(lambda x: x.split())   # This splits the movie description into individual words.

movies["tags"] = (
    movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]        #We combine all movie information into one column called tags.
)
new = movies.drop(columns=["overview", "genres", "keywords", "cast", "crew"]).reset_index(drop=True)             #Remove unnecessary columns
new["tags"] = new["tags"].apply(lambda x: " ".join(x).lower())                 #Convert tags into lowercase text                     

cv = CountVectorizer(max_features=5000, stop_words="english")                #converts text into numerical values based on word frequency.
vector = cv.fit_transform(new["tags"]).toarray()                                #This converts movie tags into numerical vectors.
similarity = cosine_similarity(vector)                                            #This converts movie tags into numerical vectors.

os.makedirs("model", exist_ok=True)                                                #This creates a folder named model.
pickle.dump(new, open("model/movie_list.pkl", "wb"))                                    #Stores movie IDs, titles and tags.
pickle.dump(similarity, open("model/similarity.pkl", "wb"))                               #Stores similarity scores between movies.
print("Done! Saved model/movie_list.pkl and model/similarity.pkl")
print("Movies:", len(new))
