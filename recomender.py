import polars as pl
import numpy as np
import lancedb

# Movie recomnder engine using vector dbs
# Data source: https://grouplens.org/datasets/movielens/latest/

# Load the Movie Lens dataset
data_folder = "data/ml-latest-small/"
# ratings: userId, movieId, rating, timestamp
df_ratings = pl.read_csv(data_folder + "ratings.csv")
# How many movies have been reviewed
nb_movies_reviewed = len(df_ratings["movieId"].unique())
# movieID, title, genres (pipe separated list of genres)
df_movies = pl.read_csv(data_folder + "movies.csv")
nb_movies_global = len(df_movies)
# movieId, imdbId, tmdbId (links from this movie to other movie dbs)
df_links = pl.read_csv(data_folder + "links.csv")
print(
    f"There are {nb_movies_global} movies in the global list. {nb_movies_reviewed} have been reviewed"
)

# Data transformation for matrix factorization
# Create a matrix where each entry (i,j) is the rating user_i gave movie_j
df_pivot = df_ratings.pivot(
    index="userId", columns="movieId", values="rating"
).fill_null(0)
print(f"Pivoted data has shape {df_pivot.shape}. One column is the index.")

all_data = df_pivot.to_numpy()
users, ratings_data = all_data[:, :1], all_data[:, 1:]
movie_ids = [int(x) for x in df_pivot.columns[1:]]

# print(ratings_data.shape)
nb_users, nb_movies = ratings_data.shape

# Compute embedding vectors by SVD
_, _, Vh = np.linalg.svd(ratings_data, full_matrices=False)
# print(Vh.shape)

# Vh contains a vector for each movie.
movie_embeds = Vh.T
embed_dim = movie_embeds.shape[1]

"""
We want to create a lancedb table with the following
1. an integer movie id field
2. a vector field of embeddings
3. a string field of genres
4. a string field for the movie title
5. an integer field for the imdb_id
"""

df_data = pl.DataFrame({"movieId": movie_ids, "vector": movie_embeds})
df_data = df_data.join(df_movies, on="movieId", how="left")
df_data = df_data.join(df_links.select(["movieId", "imdbId"]), on="movieId", how="left")

# Upload the data to the lancedb
uri = "./lancedb"
db = lancedb.connect(uri)

db.drop_table("movie_table", ignore_missing=True)
tbl = db.create_table("movie_table", data=df_data)

# Now that we have the vector db running let's run some queries.

# Let's pick a movie then use the db to find the 3 most closely related movies
idx = 11
row = df_data.slice(idx, 1)
print(row)
input_vector = row["vector"].item().to_numpy()
print(input_vector.shape)

res = (
    tbl.search(input_vector, vector_column_name="vector")
    .metric("l2")
    .limit(4)
    .to_polars()
)
print(res.select(["title", "genres", "_distance"]))
print("done")
