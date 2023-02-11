import os

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import KFold
from surprise.model_selection.validation import cross_validate
from flask import Flask, request, jsonify

app = Flask(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# The main Movies Metadata file
meta = pd.read_csv('movies_metadata.csv')
print(meta.head())
print("--------------------------------------->>")

# The subset of 100,000 ratings from 700 users on 9,000 movies
ratings = pd.read_csv('ratings_small.csv')
print(ratings.head())  # Movies in this dataset are rated out of 5 instead of 10
print("--------------------------------------->>")

# TMDb and IMDb IDs of a small subset of 9,000 movies of the Full Dataset
links = pd.read_csv('links_small.csv')
print(links.head())
print("--------------------------------------->>")

# Movie plot keywords for the MovieLens movies
keywords = pd.read_csv('keywords.csv')
print(keywords.head())
print("--------------------------------------->>")

# Cast and Crew Information for all movies in the dataset
credits = pd.read_csv('credits.csv')
print(credits.head())
print("--------------------------------------->>")

# -- Content-focused recommender --

meta['overview'] = meta['overview'].fillna('')
print(meta['overview'].head())  # Sample descriptions
print("--------------------------------------->>")

# Check the datatype of "id" in movies_metadata.csv
print(pd.DataFrame({'feature': meta.dtypes.index, 'dtype': meta.dtypes.values}))
print("--------------------------------------->>")
meta = meta.drop([19730, 29503, 35587])  # Remove these ids to solve ValueError: "Unable to parse string..."

# Convert object to int64 for compatibility during merging
meta['id'] = pd.to_numeric(meta['id'])

# Run  the following code for converting more than one value to integer
# def convert_int(x):
#     try:
#         return int(x)
#     except:
#         return np.nan

# Check the datatype of "tmdbId" in links_small.csv
print(pd.DataFrame({'feature': links.dtypes.index, 'dtype': links.dtypes.values}))
print("--------------------------------------->>")

# Convert float64 to int64
col = np.array(links['tmdbId'], np.int64)
links['tmdbId'] = col

# Merge the dataframes on column "tmdbId"
meta.rename(columns={'id': 'tmdbId'}, inplace=True)
meta = pd.merge(meta, links, on='tmdbId')
meta.drop(['imdb_id'], axis=1, inplace=True)
print(meta.head())
print("--------------------------------------->>")
# Alternatively, run the following code to reduce the size of movies_metadata.csv to match links_small.csv
# meta = meta[meta['tmdbId'].isin(links)]
# meta.shape

# Remove stop words and use TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
# Construct TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(meta['overview'])
print(tfidf_matrix.shape)
print("--------------------------------------->>")
# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Get corresponding indices of the movies

indices = pd.Series(meta.index, index=meta['original_title']).drop_duplicates()


# indicesx = pd.Series(meta.index, index=meta['genres']).drop_duplicates()

# Recommendation function
def recommend(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 15 most similar movies
    sim_scores = sim_scores[1:16]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Remove low-rated movies or outliers
    for i in movie_indices:
        pop = meta.at[i, 'vote_average']
        if pop < 5 or pop > 10:
            movie_indices.remove(i)

    # Return the most similar movies qualifying the 5.0 rating threshold
    return meta[['original_title', 'vote_average']].iloc[movie_indices]


print(recommend('Iron Man'))
print("--------------------------------------->>")
print(recommend('The Conjuring'))
print("--------------------------------------->>")
print(recommend('An American Tail'))
print("--------------------------------------->>")
print("--------------------------------------->>")
# -- User-focused recommender --

reader = Reader()  # Used to parse a file containing ratings
df = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
kf = KFold(n_splits=5)
kf.split(df)  # Split the data into folds

# Use Single Value Decomposition (SVD) for cross-validation and fitting
svd = SVD()
cross_validate(svd, df, measures=['RMSE', 'MAE'])

trainset = df.build_full_trainset()
svd.fit(trainset)

# Check a random user's ratings
print(ratings[ratings['userId'] == 10])
print("--------------------------------------->>")
# Read the smaller links file again
links_df = pd.read_csv('links_small.csv')
col = np.array(links_df['tmdbId'], np.int64)
links_df['tmdbId'] = col

# Merge movies_metadata.csv and links_small.csv files
links_df = links_df.merge(meta[['title', 'tmdbId']], on='tmdbId').set_index('title')
links_index = links_df.set_index('tmdbId')  # For label indexing


# Recommendation function

def hybrid(userId, title):
    idx = indices[title]
    tmdbId = links_df.loc[title]['tmdbId']  # Get the corresponding tmdb id

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]  # Scores of the 30 most similar movies
    movie_indices = [i[0] for i in sim_scores]

    movies = meta.iloc[movie_indices][['title', 'vote_average', 'tmdbId']]
    movies['est'] = movies['tmdbId'].apply(
        lambda x: svd.predict(userId, links_index.loc[x]['movieId']).est)  # Estimated prediction using svd
    movies = movies.sort_values('est', ascending=False)  # Rank movies according to the predicted values
    movies.columns = ['Title', 'Vote Average', 'TMDb Id', 'Estimated Prediction']
    return movies.head(15)  # Display top 15 similar movies


# Recommendations for user with id 1
print(hybrid(1, 'The Conjuring'))
print("--------------------------------------->>")
# Recommendations for user with id 30
print(hybrid(30, 'Iron Man'))
print("--------------------------------------->>")
# Recommendations for user with id 500
print(hybrid(500, 'Saturday Night Fever'))
print("--------------------------------------->>")
# Recommendations for user with id 500
print(hybrid(65, 'An American Tail'))
print("--------------------------------------->>")


@app.route('/recommend_movie', methods=['POST'])
def predict():
    userId = request.form.get('userId')
    title = request.form.get('title')

    # result = {'userId': userId, 'title': title}
    # input_query = np.array([[userId, title]])
    result = hybrid(int(userId), str(title))
    return jsonify({'Recommended_Movies': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
