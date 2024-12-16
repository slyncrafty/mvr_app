import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import math
import os

# from surprise import Dataset, Reader
# from surprise.model_selection import train_test_split
# from surprise.prediction_algorithms.knns import KNNWithZScore

# Define paths to data files
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, 'data')  # Adjust if your data directory is different
MOVIES_FILE = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
RATINGS_FILE = "https://liangfgithub.github.io/MovieData/ratings.dat?raw=true" 
RMAT_FILE = "https://d3c33hcgiwev3.cloudfront.net/I-w9Wo-HSzmUGNNHw0pCzg_bc290b0e6b3a45c19f62b1b82b1699f1_Rmat.csv?Expires=1734480000&Signature=POU53r-wt9D3qAj9LesIXs7WFzJUJyfoon7QgMqkNgXHE8rfoFoW0BGX0NwfTPp2EOhtv1BG2Ew0YRDHu2T4I5TKI2q8W-1Hn1NlNjCMr8hBWEt-cXn8PUDa-HmjkW-nPvTjDgHL2GPTRkLlMRT7-FuN1Nr2WFHW7I6IekMEsDE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A"
SIM_MATRIX_FILE = "https://media.githubusercontent.com/media/slyncrafty/TESTD/refs/heads/main/similarity_matrix_full.csv" #os.path.join(DATA_DIR, 'similarity_matrix_full.csv')
S_TOP30_FILE = os.path.join(DATA_DIR, 'S_top30.csv')
TOP10_POPULAR_FILE = os.path.join(DATA_DIR, 'top_10_popular.csv')

def load_data():
    """
    Load and preprocess the ratings and movies data.
    """
    # Load ratings
    ratings = pd.read_csv(
        RATINGS_FILE,
        sep='::',
        engine='python',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    
    # Load movies
    movies = pd.read_csv(
        MOVIES_FILE,
        sep='::',
        engine='python',
        encoding="ISO-8859-1",
        header=None,
        names=['MovieID', 'Title', 'Genres']
    )
    
    # Define "Multiple" for movies with multiple genres
    multiple_idx = pd.Series([("|" in movie) for movie in movies['Genres']])
    movies.loc[multiple_idx, 'Genres'] = 'Multiple'
    
    # Rename columns for consistency
    ratings.rename(columns={'UserID': 'userID', 'MovieID': 'movieID', 'Rating': 'rating'}, inplace=True)
    movies.rename(columns={'MovieID': 'movieID'}, inplace=True)
    
    if ratings['userID'].dtype != object:
        ratings['userID'] = ratings['userID'].astype(str)
    
    # If you want to add 'u' prefix
    if not ratings['userID'].str.startswith('u').any():
        ratings['userID'] = 'u' + ratings['userID']
    
    # Add PosterURL
    movies['PosterURL'] = "https://liangfgithub.github.io/MovieImages/" + movies['movieID'].astype(str) + ".jpg"
    
    popularity_score = ratings.groupby('movieID').size().reset_index(name='PopularityScore')
    movies = movies.merge(popularity_score, on='movieID', how='left').fillna(0)
    movies['PopularityScore'] = movies['PopularityScore'].astype(int)
    
    return ratings, movies



## System I : Recommendation Based on Popularity
## Provide the code for implementing your recommendation scheme and display the top ten movies, including their MovieID (or “m” + MovieID), title, and poster images.
def compute_top10_popular(ratings, movies, top10_path=TOP10_POPULAR_FILE, threshold=4):
    """
    Compute and save the top 10 most reviewed movies that have positive ratings (rating >= threshold).
    
    Inputs:
    - ratings: Ratings DataFrame
    - movies: Movies DataFrame
    - top10_path: Path to save 
    - threshold: Threshold for rating
    """
    # Merge ratings and movie datasets
    rating_merged = ratings.merge(movies, on='movieID', how='inner')

    # Filter the rating_merged to only keep rows where Rating >= threshold
    # Ratings are made on a 5-star scale (whole-star ratings only)
    threshold = 4
    filtered_ratings = rating_merged[rating_merged['rating'] >= threshold]

    # Count how many "positive" ratings each movie received
    movie_rating_counts = filtered_ratings.groupby('movieID').size().reset_index(name='NumPositiveRatings')

    # Merge with movie titles
    movie_counts_with_titles = movie_rating_counts.merge(movies, on='movieID', how='left')

    # Sort by number of positive ratings in descending order
    movie_counts_with_titles = movie_counts_with_titles.sort_values('NumPositiveRatings', ascending=False)

    # Select the top 10 movies based on positive rating counts
    top_10_popular = movie_counts_with_titles.head(10).copy()

    # Prefix the MovieID with 'm'
    top_10_popular['PrefixedMovieID'] = 'm' + top_10_popular['movieID'].astype(str)

    # Display the results
    display_columns = ['PrefixedMovieID', 'Title', 'NumPositiveRatings', 'PosterURL']
    top_10_popular[display_columns]

    ## Save out
    if top10_path:
        top_10_popular.to_csv(top10_path, index=False)
        
    return top_10_popular


## System II: Recommendation Based on IBCF

## Step 2: Compute the Transformed Cosine Similarity Among Movies
def compute_similarity_matrix(R_df, similarity_matrix_path=SIM_MATRIX_FILE, S_top30_path=S_TOP30_FILE):
    """
    Compute the full and top-30 similarity matrices for IBCF and save them as CSV files.
    
    Inputs:
    - R_df: Rating matrix DataFrame (users x movies)
    - similarity_matrix_path: Path to save the full similarity matrix
    - S_top30_path: Path to save the top-30 similarity matrix
    
    Output:
    - S_df
    - S_top30
    """
    ## Step 1: Normalize the Rating Matrix by Centering Each Row
    def mean_center_rows(df):
        row_means = df.mean(axis=1, skipna=True)
        centered = df.sub(row_means, axis='index')
        return centered, row_means
    
    R_centered, user_means = mean_center_rows(R_df)
    
    ## Step 2: Compute the Transformed Cosine Similarity Among Movies
    # Convert to NumPy arrays for faster computations
    mask = ~R_centered.isna().values             # shape: (num_users, num_movies)
    rating_matrix = R_centered.fillna(0).values  # NA replaced with 0
    num_users, num_movies = rating_matrix.shape
    
    # Initialize similarity matrix with NaNs
    S = np.full((num_movies, num_movies), np.nan, dtype=float)
    movie_names = R_centered.columns.tolist()
    
    # Define similarity computation function
    def compute_movie_similarity(i, j):
        common_mask = mask[:, i] & mask[:, j]
        intersection_count = np.sum(common_mask)
        if intersection_count <= 2:
            return (i, j, np.nan)
        ratings_i = rating_matrix[common_mask, i]
        ratings_j = rating_matrix[common_mask, j]
        numerator = np.sum(ratings_i * ratings_j)
        denom = math.sqrt(np.sum(ratings_i**2) * np.sum(ratings_j**2))
        if denom == 0:
            return (i, j, np.nan)
        cos_sim = numerator / denom
        transformed_sim = 0.5 + 0.5 * cos_sim
        return (i, j, transformed_sim)
    
    # Generate all upper-triangular pairs (i, j) with i < j then mirror
    pairs = [(i, j) for i in range(num_movies) for j in range(i+1, num_movies)]
    
    # Compute similarities in parallel n_jobs for # of cores
    results = Parallel(n_jobs=-1, verbose=10, backend="loky")(
        delayed(compute_movie_similarity)(i, j) for (i, j) in pairs
    )
    
    # Fill the similarity matrix
    for (i, j, val) in results:
        S[i, j] = val
        S[j, i] = val
    
    # Convert S to a DataFrame and save out
    S_df = pd.DataFrame(S, index=movie_names, columns=movie_names)
    S_df.to_csv(similarity_matrix_path)
    
    

    ## Step 3: For Each Row, Keep the Top 30 Non-NA Similarities
    ## Keep top 30 similarities per movie
    def keep_top_n(series, n=30):
        sorted_vals = series.dropna().sort_values(ascending=False)
        top_n = sorted_vals.head(n)
        out = pd.Series(np.nan, index=series.index)
        out[top_n.index] = top_n
        return out
    
    S_top30 = S_df.apply(lambda row: keep_top_n(row, n=30), axis=1)
    S_top30.to_csv(S_top30_path)
    
    return S_df, S_top30



def load_similarity_matrices(similarity_matrix_path=SIM_MATRIX_FILE, S_top30_path=S_TOP30_FILE):
    """
    Load the full and top-30 similarity matrices from CSV files.
    
    Output:
    - S_df: Full similarity matrix DataFrame
    - S_top30_df: Top-30 similarity matrix DataFrame
    """
    S_df = pd.read_csv(similarity_matrix_path, index_col=0)
    S_top30_df = pd.read_csv(S_top30_path, index_col=0)
    return S_df, S_top30_df


def load_top10_popular(top10_path=TOP10_POPULAR_FILE):
    """
    Load the top 10 popular movies from CSV.
    
    Output:
    - top_10_popular: DataFrame containing top 10 popular movies
    """
    
    top_10_popular = pd.read_csv(top10_path)
    return top_10_popular



def prepare_recommendation_system():
    """
    Prepare all necessary data and matrices for the recommendation systems.
    
    Output:
    - ratings: Ratings DataFrame
    - movies: Movies DataFrame
    - S_top30: Top-30 similarity matrix DataFrame
    - top_10_popular: Top 10 popular movies DataFrame
    """
    ratings, movies = load_data()
    
    # Check if the files exist in the directory
    if os.path.exists(TOP10_POPULAR_FILE):
        top_10_popular = load_top10_popular()
    else:
        top_10_popular = compute_top10_popular(ratings, movies)

        
    if not (os.path.exists(SIM_MATRIX_FILE) or os.path.exists(S_TOP30_FILE)):
        # Load Rmat.csv
        R_df = pd.read_csv(RMAT_FILE, index_col=0)
        # Compute similarity matrices
        S_df, S_top30 = compute_similarity_matrix(R_df)
    else:
        # Load existing files
        S_df, S_top30 = load_similarity_matrices()
    
    target_movies = ["m1", "m10", "m100", "m1510", "m260", "m3212"]
    # Subset S to these rows and columns
    subset_S = S_df.loc[target_movies, target_movies]
    # Round to 7 decimal places
    subset_S_rounded = subset_S.round(7) ## decimal places fix
    pd.set_option('display.float_format', '{:.7f}'.format) 
    print("7 x 7 similarity matrix...")
    print(subset_S_rounded)
    
    return ratings, movies, S_top30, top_10_popular


## Step 4: Create the myIBCF Function
def myIBCF(newuser, S_top30_df, popularity_ranking_df, n_recommend=10):
    """
    Generate top N movie recommendations for a new user based on Item-Based Collaborative Filtering.
    
    Inputs:
    - newuser: pd.Series indexed by 'm1', 'm2', ..., 'm3706' containing user ratings.
    - S_top30_df: DataFrame containing top 30 similarities per movie.
    - popularity_ranking_df: DataFrame containing top 10 movies
    - n_recommend: Number of recommendations to generate.
    
    Output:
    - List of recommended movieIDs w/ 'm' prefix.
    """
    rated_mask = newuser.notna()
    rated_movies = newuser.index[rated_mask]
    unrated_movies = newuser.index[~rated_mask]

    predictions = pd.Series(index=unrated_movies, dtype=float)

    for m_i in unrated_movies:
        # Check if the movie exists in the similarity matrix
        if m_i not in S_top30_df.index:
            continue  # Skip movies without similarity data
        
        neighbors = S_top30_df.loc[m_i].dropna().index  # Movies with known similarity to m_i
        # Only consider neighbors that the new user has rated
        rated_neighbors = neighbors.intersection(rated_movies)

        if len(rated_neighbors) > 0:
            # Similarities for these neighbors
            sim_values = S_top30_df.loc[m_i, rated_neighbors]
            user_ratings = newuser[rated_neighbors]

            denom = sim_values.sum()
            if denom == 0:
                pred = np.nan
            else:
                num = (sim_values * user_ratings).sum()
                pred = num / denom
        else:
            pred = np.nan

        predictions[m_i] = pred

    # Select top N based on predictions
    pred_sorted = predictions.dropna().sort_values(ascending=False)
    recommended = pred_sorted.head(n_recommend)

    if len(recommended) < n_recommend:
        # Add popular ranking to fill up spots
        # Exclude not rated or already recommended
        already_rated = set(newuser.index[rated_mask])
        already_recommended = set(recommended.index)
        additional_needed = n_recommend - len(recommended)
        
        # Filter out already rated and already recommended movies from popularity ranking
        pop_candidates = popularity_ranking_df[
            ~popularity_ranking_df['PrefixedMovieID'].isin(already_rated | already_recommended)
        ]
        pop_fill = pop_candidates.head(additional_needed)['PrefixedMovieID'].tolist()
        
        # Append [np.nan]*len(pop_fill) indexed by 'mXXXX' strings
        if pop_fill:
            pop_fill_series = pd.Series([np.nan]*len(pop_fill), index=pop_fill)
            recommended = pd.concat([recommended, pop_fill_series])

    # Return list of movieIDs with 'm' prefix
    return recommended.index.tolist()



def get_recommended_movies(recommended_movie_ids, movies_df, predicted_ratings=None):
    """
    Generate a DataFrame of recommended movies with titles and poster URLs.
    
    Inputs:
    - recommended_movie_ids: List of movieIDs with 'm' prefix
    - movies_df: Movies DataFrame
    - predicted_ratings: Optional list or Series of predicted ratings
    
    Output:
    - recommended_df: DataFrame containing recommended movies details
    """
    print("Recommended Movie IDs:", recommended_movie_ids)  # Debugging
    
    # Convert 'mXXXX' to integer movieID
    recommended_df = pd.DataFrame(recommended_movie_ids, columns=['PrefixedMovieID'])
    
    #unique_ids = recommended_df['PrefixedMovieID'].unique()
    #print("Unique PrefixedMovieIDs:", unique_ids)
    
    recommended_df['movieID'] = recommended_df['PrefixedMovieID'].apply(lambda x: int(x[1:]))
    
    # Merge with movies to get titles and poster URLs
    recommended_df = recommended_df.merge(movies_df, on='movieID', how='left')
    
    
    if predicted_ratings is not None and len(predicted_ratings) == len(recommended_df):
        recommended_df['PredictedRating'] = predicted_ratings
    else:
        recommended_df['PredictedRating'] = np.nan
    
    return recommended_df[['PrefixedMovieID', 'Title', 'PredictedRating', 'PosterURL']]



def get_displayed_movies(movies_df, n=100, popular_ratio=0.7, random_seed=82):
    """
    Select a subset of movies to display for rating.
    
    Inputs:
    - movies_df: Movies DataFrame
    - n: Number of movies to display
    - popular_ratio: Fraction selected based on popular score
    - random_seed: random seed for reproducibility
    
    Output:
    - displayed_movies: DataFrame containing selected movies
    """
    #to_display = movies_df.sample(n=n, random_state=42).reset_index(drop=True)
    #to_display = movies_df.sort_values(by='PopularityScore', ascending=False).head(n)
    n_popular = int(n * popular_ratio)
    n_random = n - n_popular
    
    # Select top popular movies
    top_popular = movies_df.sort_values(by='PopularityScore', ascending=False).head(n_popular)
    
    # Select random movies from the remaining pool
    remaining_movies = movies_df.drop(top_popular.index)
    if n_random > 0:
        random_movies = remaining_movies.sample(n=n_random, random_state=random_seed)
    else:
        random_movies = pd.DataFrame()
    
    # Combine and shuffle
    to_display = pd.concat([top_popular, random_movies]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"Selected {len(to_display)} movies: {n_popular} popular and {n_random} random.")
    
    return to_display
