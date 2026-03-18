"""
tmdb_api.py — TMDb API Helper

Fetches movie posters, ratings, and details from The Movie Database (TMDb) API.
"""

import requests

TMDB_API_KEY = "1cd9980bcf2b89af9e5689a000761672"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def get_movie_details(title: str) -> dict:
    """
    Search for a movie on TMDb and return its poster, rating, overview, and year.

    Parameters:
        title (str): Movie title to search for.

    Returns:
        dict: Movie details with keys: poster_url, rating, year, overview
              Returns defaults if not found.
    """
    defaults = {
        "poster_url": None,
        "rating": None,
        "year": None,
        "overview": None,
    }

    try:
        # Search for the movie by title
        response = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title},
            timeout=5,
        )
        data = response.json()

        if data.get("results"):
            movie = data["results"][0]  # Take the top result

            poster_path = movie.get("poster_path")
            poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None

            release_date = movie.get("release_date", "")
            year = release_date[:4] if release_date else None

            return {
                "poster_url": poster_url,
                "rating": movie.get("vote_average"),
                "year": year,
                "overview": movie.get("overview"),
            }
    except Exception:
        pass

    return defaults
