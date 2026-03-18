"""
recommender.py — Movie Recommendation Engine

This is the BRAIN of the project! It contains the logic that:
  1. Converts movie text descriptions into numerical vectors
  2. Computes how similar movies are to each other
  3. Finds and returns the top 5 most similar movies

TWO APPROACHES are implemented:
  - CountVectorizer: Counts how many times each word appears (simpler)
  - TF-IDF: Weighs words by importance — rare words get higher scores (smarter)

KEY CONCEPT — How Content-Based Filtering Works:
  Imagine each movie as a point in a high-dimensional space.
  Movies with similar descriptions are close together.
  We measure "closeness" using COSINE SIMILARITY (angle between vectors).
  Cosine similarity = 1 means identical, 0 means completely different.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class MovieRecommender:
    """
    A class that builds and manages the movie recommendation engine.

    WHY USE A CLASS?
      A class bundles data (the DataFrame, the similarity matrix) with
      functions (recommend, search). This keeps everything organized and
      makes it easy to switch between CountVectorizer and TF-IDF.

    Usage:
        recommender = MovieRecommender(df)
        recommender.build_model(method="tfidf")
        results = recommender.recommend("Avatar")
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the recommender with a preprocessed DataFrame.

        Parameters:
            df (pd.DataFrame): Must have 'title' and 'tags' columns.
                               Get this from data_loader.get_prepared_data()
        """
        self.df = df  # Store the movie data
        self.similarity_matrix = None  # Will be computed when build_model() is called
        self.method = None  # Track which method was used

    def build_model(self, method: str = "count") -> None:
        """
        Build the similarity matrix using the chosen vectorization method.

        This is the MOST IMPORTANT function in the project!

        HOW IT WORKS:
          Step 1: Convert text → numbers (vectorization)
            - Each movie's 'tags' text becomes a vector of numbers
            - "action adventure space" might become [1, 1, 0, 0, 1, ...]
          Step 2: Compute cosine similarity between ALL pairs of movies
            - Result is a matrix where entry [i][j] = similarity between movie i and j

        Parameters:
            method (str): Either "count" for CountVectorizer or "tfidf" for TF-IDF.
        """
        self.method = method

        if method == "tfidf":
            # ════════════════════════════════════════════════
            # TF-IDF (Term Frequency - Inverse Document Frequency)
            # ════════════════════════════════════════════════
            # TF  = How often a word appears in a document (movie description)
            # IDF = How rare a word is across ALL documents
            # TF-IDF = TF × IDF
            #
            # WHY IS THIS BETTER?
            #   Common words like "movie" or "story" appear in almost every overview.
            #   TF-IDF gives them LOW scores because they're not useful for distinguishing.
            #   Rare words like "superhero" or "spacecraft" get HIGH scores because
            #   they actually tell us what makes a movie unique.
            #
            # max_features=5000 → only keep the 5000 most important words
            # stop_words='english' → remove common English words (the, is, at, etc.)
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            print("[INFO] Using TF-IDF Vectorizer (smarter, weights words by importance)")

        else:
            # ════════════════════════════════════════════════
            # CountVectorizer (Bag of Words)
            # ════════════════════════════════════════════════
            # Simply counts how many times each word appears in each movie's tags.
            # "action action adventure" → action=2, adventure=1
            #
            # This is simpler but still works well for this project!
            #
            # max_features=5000 → only keep the 5000 most frequent words
            # stop_words='english' → remove common English words
            vectorizer = CountVectorizer(max_features=5000, stop_words="english")
            print("[INFO] Using CountVectorizer (counts word occurrences)")

        # --- Step 1: Fit and Transform ---
        # fit_transform() does two things:
        #   fit = learn the vocabulary (which words exist)
        #   transform = convert each movie's tags into a number vector
        # The result is a SPARSE MATRIX (saves memory by not storing zeros)
        vectors = vectorizer.fit_transform(self.df["tags"])
        print(f"[INFO] Created vectors of shape: {vectors.shape}")
        print(f"   -> {vectors.shape[0]} movies, {vectors.shape[1]} word features")

        # --- Step 2: Compute Cosine Similarity ---
        # cosine_similarity() compares every movie with every other movie
        # Result: a matrix of size (num_movies × num_movies)
        # similarity_matrix[i][j] = how similar movie i is to movie j
        # The diagonal is always 1.0 (a movie is identical to itself)
        self.similarity_matrix = cosine_similarity(vectors)
        print(f"[OK] Similarity matrix computed: {self.similarity_matrix.shape}")

    def recommend(self, movie_name: str, top_n: int = 5) -> list:
        """
        Recommend movies similar to the given movie name.

        Parameters:
            movie_name (str): The name of the movie to find recommendations for.
            top_n (int): How many recommendations to return (default: 5).

        Returns:
            list: A list of dictionaries, each with 'title' and 'score' keys.
                  Returns an empty list if the movie is not found.
                  Returns a special dict with 'suggestions' key for partial matches.
        """
        # Make sure the model has been built first
        if self.similarity_matrix is None:
            print("[WARNING] Model not built yet! Call build_model() first.")
            return []

        # --- Case-Insensitive Matching ---
        # Convert the user's input to lowercase and compare with lowercase titles
        # This way "avatar", "Avatar", "AVATAR" all work the same
        movie_name_lower = movie_name.strip().lower()

        # Create a lowercase version of all titles for comparison
        titles_lower = self.df["title"].str.lower()

        # Try to find an exact match (case-insensitive)
        matches = self.df[titles_lower == movie_name_lower]

        if len(matches) == 0:
            # --- No Exact Match → Try Partial Matching ---
            # If user types "aveng", find all titles containing "aveng"
            # .str.contains() checks if the substring exists in each title
            partial_matches = self.df[titles_lower.str.contains(movie_name_lower, na=False)]

            if len(partial_matches) > 0:
                # Found partial matches! Return them as suggestions
                suggestions = partial_matches["title"].tolist()[:10]  # Limit to 10
                return [{"suggestions": suggestions}]
            else:
                # No matches at all
                return []

        # Get the INDEX of the matched movie (first match if duplicates exist)
        movie_index = matches.index[0]

        # --- Get Similarity Scores ---
        # self.similarity_matrix[movie_index] is a 1D array with the similarity
        # score between our movie and EVERY other movie
        similarity_scores = self.similarity_matrix[movie_index]

        # --- Sort by Similarity (Highest First) ---
        # enumerate() pairs each score with its index: [(0, 0.1), (1, 0.9), ...]
        # sorted() sorts by the score (key=lambda x: x[1])
        # reverse=True means highest scores first
        # [1:top_n+1] skips index 0 (the movie itself, which has score 1.0)
        scored_movies = sorted(
            enumerate(similarity_scores),
            key=lambda x: x[1],
            reverse=True
        )[1:top_n + 1]  # Skip first (the movie itself)

        # --- Build the Results List ---
        recommendations = []
        for idx, score in scored_movies:
            recommendations.append({
                "title": self.df.iloc[idx]["title"],  # Movie name
                "score": round(score, 4)               # Similarity score (0 to 1)
            })

        return recommendations

    def get_movie_count(self) -> int:
        """Return the total number of movies in the dataset."""
        return len(self.df)


# ============================================================
# TESTING — Run this file directly to test the recommender
# ============================================================
if __name__ == "__main__":
    from data_loader import get_prepared_data

    print("=" * 60)
    print("  RECOMMENDER — Testing Mode")
    print("=" * 60)

    # Load and preprocess data
    df = get_prepared_data()

    # Create recommender and build model
    rec = MovieRecommender(df)
    rec.build_model(method="tfidf")

    # Test with a known movie
    print("\n" + "=" * 60)
    print("  Testing: recommend('Avatar')")
    print("=" * 60)
    results = rec.recommend("Avatar")
    for i, movie in enumerate(results, 1):
        print(f"  {i}. {movie['title']} (similarity: {movie['score']})")
