import requests

key = "8eb8afb96cff6762b955b91be75c7b19"
r = requests.get(f"https://api.themoviedb.org/3/movie/19995?api_key={key}", timeout=10)
print(r.status_code)
print(r.text[:300])