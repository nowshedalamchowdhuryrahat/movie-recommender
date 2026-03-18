"""
data_loader.py — Load and Preprocess the TMDB Movie Dataset

This module handles:
  1. Loading the CSV file into a pandas DataFrame
  2. Selecting only the columns we need (title, overview, genres)
  3. Parsing the 'genres' column from JSON strings into readable text
  4. Handling missing values
  5. Creating a combined 'tags' column for the recommendation engine

WHY IS THIS A SEPARATE FILE?
  Keeping data loading in its own module makes the code organized and reusable.
  If you later switch to a different dataset, you only change this file.
"""

import pandas as pd
import json
import os


def load_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Load the TMDB 5000 movies CSV file.

    Parameters:
        filepath (str): Path to the CSV file. If not provided, defaults to
                        'data/tmdb_5000_movies.csv' in the project directory.

    Returns:
        pd.DataFrame: Raw DataFrame loaded from the CSV.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist at the given path.
    """
    # If no path is given, use the default location
    if filepath is None:
        # os.path.dirname(__file__) gives the folder where THIS script lives
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, "data", "tmdb_5000_movies.csv")

    # Check if the file exists before trying to load it
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {filepath}\n"
            f"Please download 'tmdb_5000_movies.csv' from Kaggle:\n"
            f"   https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
            f"   and place it in the 'data/' folder.\n"
        )

    # Read the CSV file into a DataFrame
    # A DataFrame is like a spreadsheet / table in Python
    df = pd.read_csv(filepath)
    print(f"[OK] Dataset loaded successfully! ({len(df)} movies)")
    return df


def extract_genre_names(genres_json_string: str) -> str:
    """
    Convert the genres column from JSON format to a simple space-separated string.

    The genres column in the CSV looks like this:
        [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]

    We want to convert it to:
        "Action Adventure"

    Parameters:
        genres_json_string (str): The raw JSON string from the genres column.

    Returns:
        str: Space-separated genre names (e.g., "Action Adventure Sci-Fi").
    """
    try:
        # json.loads() converts a JSON string into a Python list of dictionaries
        genres_list = json.loads(genres_json_string)

        # Extract just the 'name' from each dictionary in the list
        genre_names = [genre["name"] for genre in genres_list]

        # Join them with spaces (not commas) so CountVectorizer treats each as a word
        return " ".join(genre_names)

    except (json.JSONDecodeError, TypeError, KeyError):
        # If anything goes wrong (bad JSON, missing key, etc.), return empty string
        # This prevents the whole program from crashing on bad data
        return ""


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the DataFrame for the recommendation engine.

    Steps:
      1. Keep only the columns we need: 'title', 'overview', 'genres'
      2. Handle missing values (fill NaN with empty strings)
      3. Parse genres from JSON to plain text
      4. Create a combined 'tags' column = overview + genres
      5. Convert tags to lowercase for consistent matching

    Parameters:
        df (pd.DataFrame): The raw DataFrame from load_dataset().

    Returns:
        pd.DataFrame: Cleaned DataFrame with a 'tags' column ready for vectorization.
    """
    # --- Step 1: Select only the columns we need ---
    # We only need title (to identify movies), overview (description), and genres
    # Using .copy() to avoid the SettingWithCopyWarning from pandas
    df = df[["title", "overview", "genres"]].copy()
    print(f"[INFO] Selected columns: title, overview, genres")

    # --- Step 2: Handle missing values ---
    # Some movies might have no overview or no genres listed
    # fillna("") replaces NaN (Not a Number / missing) with an empty string
    # This prevents errors when we try to combine text later
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("[]")  # Empty JSON array for missing genres
    print(f"[INFO] Handled missing values")

    # --- Step 3: Parse genres from JSON strings to plain text ---
    # The .apply() method runs a function on every row in a column
    # Here, we apply extract_genre_names to each entry in the 'genres' column
    df["genres"] = df["genres"].apply(extract_genre_names)
    print(f"[INFO] Parsed genres into readable text")

    # --- Step 4: Create the combined 'tags' column ---
    # We combine overview and genres into a single text field
    # WHY? Because the recommendation engine needs ONE text column to compare movies
    # Example: "A marine on an alien planet... Action Adventure Science Fiction"
    df["tags"] = df["overview"] + " " + df["genres"]
    print(f"[INFO] Created 'tags' column (overview + genres)")

    # --- Step 5: Convert to lowercase ---
    # This ensures "Action" and "action" are treated the same
    # Without this, the vectorizer might count them as different words
    df["tags"] = df["tags"].str.lower()
    print(f"[INFO] Converted tags to lowercase")

    # --- Final check: Drop any rows where title is missing ---
    # A movie without a title is useless to us
    df = df.dropna(subset=["title"])

    # Reset the index so it goes 0, 1, 2, ... without gaps
    # WHY? Because we use the index to look up movies in the similarity matrix
    df = df.reset_index(drop=True)

    print(f"[OK] Preprocessing complete! ({len(df)} movies ready)")
    return df


def get_prepared_data(filepath: str = None) -> pd.DataFrame:
    """
    Convenience function that loads AND preprocesses the data in one call.

    This is the main function you'll use from other files:
        from data_loader import get_prepared_data
        df = get_prepared_data()

    Parameters:
        filepath (str): Optional path to the CSV file.

    Returns:
        pd.DataFrame: Fully cleaned and preprocessed DataFrame.
    """
    raw_df = load_dataset(filepath)
    clean_df = preprocess_data(raw_df)
    return clean_df


# ============================================================
# WHAT HAPPENS IF YOU RUN THIS FILE DIRECTLY?
# The code below only runs when you execute: python data_loader.py
# It does NOT run when you import this file from another script.
# This is useful for testing!
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DATA LOADER — Testing Mode")
    print("=" * 60)

    df = get_prepared_data()

    # Show the first 5 rows to verify everything looks right
    print("\nFirst 5 movies in the dataset:\n")
    print(df[["title", "tags"]].head().to_string(index=False))

    print("\nDataset shape:", df.shape)
    print("Columns:", list(df.columns))
