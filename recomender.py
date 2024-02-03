import polars as pl

# Movie recomnder engine using vector dbs

# https://grouplens.org/datasets/movielens/latest/
data_folder = "data/ml-latest-small/"
# ratings: userId, movieId, rating, timestamp
df_ratings = pl.read_csv(data_folder + "ratings.csv")
print(df_ratings)
# create a matrix where each entry (i,j) is the rating user_i gave movie_j
df_pivot = df_ratings.pivot(
    index="userId", columns="movieId", values="rating"
).fill_null(0)
print(df_pivot.head())
