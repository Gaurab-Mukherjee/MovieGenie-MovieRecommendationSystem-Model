import json
import time
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# API key for the TMDB API
api_key = "c1b5b5d1017cbf9f1ae2e311e9ab068a"

count = 0


def retrieve_movies(year, api_key):
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}&language=en-US&with_original_language=hi&sort_by=popularity.desc&include_adult=false&include_video=false&page=1&primary_release_year={year}"
    response = requests.get(url)
    return response.json()


def pre_process_data(movies):
    movie_data = []
    for movie in movies["results"]:
        movie_info = {}
        movie_info["id"] = movie["id"]
        movie_data.append(movie_info)
    return movie_data


# Get the current year
current_year = int(time.strftime("%Y"))

# Initialize a list to store movie data
all_movies = []

# Retrieve movie data for each year in the last 50 years
for year in range(current_year, current_year - 50, -1):
    movies = retrieve_movies(year, api_key)
    movie_data = pre_process_data(movies)
    all_movies += movie_data


# Function to retrieve movie data from the TMDB API
def retrieve_movie_details(movie_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    return response.json()


# Function to retrieve the TMDB ID and IMDB ID for a given movie ID
def get_tmdb_and_imdb_id(movie_id, api_key):
    movie_details = retrieve_movie_details(movie_id, api_key)
    tmdb_id = movie_details["id"]
    imdb_id = movie_details["imdb_id"]
    return tmdb_id, imdb_id


all_list = []

for id_movie in all_movies:
    movie_id = id_movie
    tmdb_id, imdb_id = get_tmdb_and_imdb_id(movie_id["id"], api_key)
    count = count + 1
    if imdb_id is not None:
        imdbID = imdb_id[3:]
    else:
        imdbID = ""
    info = [{"movieId": count, "imdbId": imdbID, "tmdbId": tmdb_id}]
    all_list += info

links = pd.DataFrame(all_list)


# Function to pre-process the movie data
def meta_process_data(movies):
    movie_meta_data = []
    for movie in movies["results"]:
        movie_info = {}
        movie_info["adult"] = movie["adult"]
        movie_info["backdrop_path"] = movie["backdrop_path"]
        movie_info["genre_ids"] = movie["genre_ids"]
        movie_info["id"] = movie["id"]
        movie_info["original_language"] = movie["original_language"]
        movie_info["original_title"] = movie["original_title"]
        movie_info["overview"] = movie["overview"]
        movie_info["popularity"] = movie["popularity"]
        movie_info["poster_path"] = movie["poster_path"]
        movie_info["release_date"] = movie["release_date"]
        movie_info["title"] = movie["title"]
        movie_info["video"] = movie["video"]
        movie_info["vote_average"] = movie["vote_average"]
        movie_info["vote_count"] = movie["vote_count"]
        movie_meta_data.append(movie_info)
    return movie_meta_data


# Initialize a list to store movie data
all_movies_meta = []

# Retrieve movie data for each year in the last 50 years
for year in range(current_year, current_year - 50, -1):
    movies = retrieve_movies(year, api_key)
    movie_meta_data = meta_process_data(movies)
    all_movies_meta += movie_meta_data

meta = pd.DataFrame(all_movies_meta)
meta = meta[
    ['adult', 'backdrop_path', 'genre_ids', 'id', 'original_language', 'original_title', 'overview', 'popularity',
     'poster_path', 'release_date', 'title', 'video', 'vote_average', 'vote_count']]
meta['overview'] = meta['overview'].fillna('')
meta['poster_path'] = 'https://image.tmdb.org/t/p/w500' + meta['poster_path']
# # Convert object to int64 for compatibility during merging
meta['id'] = pd.to_numeric(meta['id'])
# Convert float64 to int64
col = np.array(links['tmdbId'], np.int64)
links['tmdbId'] = col

# Merge the dataframes on column "tmdbId"
meta.rename(columns={'id': 'tmdbId'}, inplace=True)
meta = pd.merge(meta, links, on='tmdbId')
# Remove stop words and use TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
# Construct TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(meta['overview'])
# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Get corresponding indices of the movies
indices = pd.Series(meta.index, index=meta['title']).drop_duplicates()
indicesx = pd.Series(meta.index, index=meta['genre_ids']).drop_duplicates()


# Recommendation function
def recommend(title, cosine_sim=cosine_sim, meta=meta):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 15 most similar movies
    sim_scores = sim_scores[1:12]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Remove low-rated movies or outliers
    for i in movie_indices:
        pop = meta.at[i, 'vote_average']
        if pop < 5 or pop > 10:
            movie_indices.remove(i)
    # Return the most similar movies qualifying the 5.0 rating threshold
    # return {'results': meta[['adult', 'backdrop_path', 'genre_ids', 'tmdbId', 'original_language', 'original_title',
    #                          'overview',
    #                          'popularity',
    #                          'poster_path', 'release_date', 'title', 'video', 'vote_average', 'vote_count', 'movieId',
    #                          'imdbId']].iloc[movie_indices].to_dict('records')}

    data = {'results': meta[['adult', 'backdrop_path', 'genre_ids', 'tmdbId', 'original_language', 'original_title',
                             'overview',
                             'popularity',
                             'poster_path', 'release_date', 'title', 'video', 'vote_average', 'vote_count', 'movieId',
                             'imdbId']].iloc[movie_indices].to_dict('records')}

    json_data = json.dumps(data, ensure_ascii=False).encode('utf8')
    return json_data


# Emoji Function
def emoji_recommend(emoji_keyword):
    # Set up dictionary of emoji keywords and corresponding TMDB genres
    api_key = "c1b5b5d1017cbf9f1ae2e311e9ab068a"
    emoji_dict = {
        "🎬": 28,  # Action
        "🎭": 18,  # Drama
        "🤡": 35,  # Comedy
        "👻": 27,  # Horror
        "🧝‍♀️": 14,  # Fantasy
        "🚀": 878,  # Science Fiction
        "💘": 10749,  # Romance
        "👑": 14,  # Fantasy
        "🕵️": 9648,  # Mystery
        "👨‍👩‍👧‍👦": 10751,  # Family
        "😈": 80,  # Crime
        "🪂": 12,  # Adventure
        "🏺": 36  # History
    }

    # Choose a random emoji keyword and corresponding genre
    # emoji_keyword, genre = random.choice(list(emoji_dict.items()))
    # emoji_keyword = input("Enter an emoji: ")
    genre = emoji_dict[emoji_keyword]

    # Set up TMDB API request for movies in chosen genre
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}&with_genres={genre}&language=en-US&with_original_language=hi"

    # Send request and retrieve list of movies
    response = requests.get(url)
    movies = response.json()["results"]
    recommendations = []
    for i in range(10):
        # Choose a random movie from the list
        chosen_movie = random.choice(movies)
        recommendations.append(chosen_movie)
    # Return recommendation data
    return {'results': recommendations}


################################################################################
# ========================= Genre based recommendation system ==================
################################################################################

def get_movie_genres(year):
    url = f'https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}&language=en-US&with_original_language=hi&page=1&primary_release_year={year}'
    response = requests.get(url)
    data = json.loads(response.text)
    genres = {genre['name']: genre['id'] for genre in data['genres']}
    return genres


def get_movie_ids_by_genre(genre_id, year):
    url = f'https://api.themoviedb.org/3/discover/movie?api_key={api_key}&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&page=1&with_genres={genre_id}&with_original_language=hi&page=1&primary_release_year={year}'
    response = requests.get(url)
    data = json.loads(response.text)
    movie_ids = [movie['id'] for movie in data['results']]
    return random.sample(movie_ids, k=10)


def get_movie_details(movie_id, year):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US&with_original_language=hi&page=1&primary_release_year={year}'
    response = requests.get(url)
    data = json.loads(response.text)
    return data


def recommend_movies_by_genre(genre, year):
    genres = get_movie_genres(year)
    if genre not in genres:
        return 'Invalid genre'
    genre_id = genres[genre]
    movie_ids = get_movie_ids_by_genre(genre_id, year)
    recommendations = []
    for movie_id in movie_ids[:10]:
        movie_details = get_movie_details(movie_id, year)
        recommendations.append(movie_details)
    return {'results': recommendations}


# recommend_movies_by_genre('Romance', 2022)


@app.route('/recommend_movie', methods=['POST'])
def predict():
    request_data = request.data
    json_data = json.loads(request_data)
    title = json_data['title']
    result = recommend(title)
    return result


@app.route('/emoji_recommend_movie', methods=['POST'])
def predict_emoji():
    emoji = request.form.get('emoji')
    result = emoji_recommend(emoji)
    return result


@app.route('/genre_recommend_movie', methods=['POST'])
def predict_genre():
    genre = request.form.get('genre')
    year = request.form.get('year')
    result = recommend_movies_by_genre(genre, year)
    return result


if __name__ == '__main__':
    app.run(debug=True)
