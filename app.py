"""
app.py — Flask Web Application for Movie Recommendations

This file turns your recommendation system into a WEB APP!
Instead of typing in the terminal, users can open a browser,
type a movie name in a search box, and see recommendations on a webpage.

WHAT IS FLASK?
  Flask is a lightweight Python web framework. It lets you create websites
  and web APIs with just a few lines of code. It's one of the most popular
  frameworks for beginners.

HOW TO RUN:
  python app.py
  Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request
from data_loader import get_prepared_data
from recommender import MovieRecommender
from tmdb_api import get_movie_details

# ============================================================
# Create the Flask app
# __name__ tells Flask where to find templates and static files
# ============================================================
app = Flask(__name__)

# ============================================================
# Global variables to store our data and model
# We load them ONCE when the server starts, not on every request
# This makes the app fast because we don't re-read the CSV each time
# ============================================================
recommender = None


def initialize_model():
    """
    Load the dataset and build the recommendation model.
    Called once when the server starts.
    """
    global recommender

    print("\n>> Initializing Movie Recommendation Engine...")
    try:
        df = get_prepared_data()
        recommender = MovieRecommender(df)
        recommender.build_model(method="tfidf")  # Using TF-IDF for best results
        print(f"[OK] Model ready! {recommender.get_movie_count()} movies loaded.\n")
    except FileNotFoundError as e:
        print(e)
        print("[WARNING] The web app will start but recommendations won't work.")
        print("   Please add the dataset and restart the server.\n")


@app.route("/", methods=["GET", "POST"])
def home():
    """
    The main (and only) route of our web app.

    GET request  → Show the search page (empty)
    POST request → Process the search and show results

    WHAT IS A ROUTE?
      A route maps a URL to a Python function.
      @app.route("/") means: when someone visits the homepage,
      run the home() function.

    WHAT ARE GET AND POST?
      GET  = "Show me the page" (loading a page)
      POST = "I'm submitting data" (submitting a form)
    """
    recommendations = None
    suggestions = None
    error = None
    searched_movie = ""

    if request.method == "POST":
        # Get the movie name from the form input
        # request.form["movie_name"] reads the value of the input field
        searched_movie = request.form.get("movie_name", "").strip()

        if not searched_movie:
            error = "Please enter a movie name."
        elif recommender is None:
            error = "Model not loaded. Please add the dataset and restart the server."
        else:
            # Get recommendations from our engine
            results = recommender.recommend(searched_movie)

            if not results:
                error = f"Movie '{searched_movie}' not found. Try a different name."
            elif "suggestions" in results[0]:
                # Partial match — show suggestions
                suggestions = results[0]["suggestions"]
            else:
                # Success! Enrich with TMDb posters & ratings
                for movie in results:
                    details = get_movie_details(movie["title"])
                    movie["poster_url"] = details["poster_url"]
                    movie["rating"] = details["rating"]
                    movie["year"] = details["year"]
                recommendations = results

    # render_template() loads the HTML file from the 'templates/' folder
    # and passes our Python variables to it (so the HTML can display them)
    return render_template(
        "index.html",
        recommendations=recommendations,
        suggestions=suggestions,
        error=error,
        searched_movie=searched_movie,
    )


# ============================================================
# Start the Flask development server
# debug=True means the server auto-reloads when you change code
# ============================================================
if __name__ == "__main__":
    initialize_model()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    # use_reloader=False prevents the model from being loaded twice
