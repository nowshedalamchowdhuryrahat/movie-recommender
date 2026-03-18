"""
main.py — Command-Line Interface (CLI) Entry Point

This is the file you run to use the Movie Recommendation System from the terminal:
    python main.py

It provides an interactive loop where you can:
  - Type a movie name to get recommendations
  - Choose between CountVectorizer or TF-IDF method
  - See similarity scores alongside movie names
  - Type 'quit' to exit

WHAT IS A CLI?
  CLI = Command Line Interface. It means you interact through the terminal/console
  by typing text, instead of clicking buttons in a graphical interface.
"""

from data_loader import get_prepared_data
from recommender import MovieRecommender


def display_banner():
    """Print a nice welcome banner when the program starts."""
    print("\n" + "═" * 60)
    print("  🎬  MOVIE RECOMMENDATION SYSTEM  🎬")
    print("  Content-Based Filtering with Python")
    print("═" * 60)


def choose_method() -> str:
    """
    Let the user choose which vectorization method to use.

    Returns:
        str: Either "count" or "tfidf"
    """
    print("\n📊 Choose a recommendation method:")
    print("  1. CountVectorizer (simple word counting)")
    print("  2. TF-IDF (smarter, weights words by importance) ⭐ Recommended")
    print()

    while True:
        choice = input("Enter 1 or 2 (default=2): ").strip()

        if choice == "1":
            return "count"
        elif choice == "2" or choice == "":
            return "tfidf"
        else:
            print("  ⚠️  Please enter 1 or 2.")


def display_recommendations(movie_name: str, results: list):
    """
    Display the recommendation results in a nicely formatted way.

    Parameters:
        movie_name (str): The movie the user searched for.
        results (list): List of recommendation dictionaries from the recommender.
    """
    if not results:
        # No results at all — movie not found
        print(f"\n  ❌ Movie '{movie_name}' not found in the dataset.")
        print("  💡 Try a different spelling or a partial name (e.g., 'dark knight').\n")
        return

    # Check if the result is a list of suggestions (partial match)
    if "suggestions" in results[0]:
        suggestions = results[0]["suggestions"]
        print(f"\n  🔍 No exact match for '{movie_name}'. Did you mean:")
        for i, title in enumerate(suggestions, 1):
            print(f"     {i}. {title}")
        print(f"\n  💡 Try typing the full movie name.\n")
        return

    # Display the recommendations!
    print(f"\n  🎯 Top {len(results)} movies similar to '{movie_name}':")
    print("  " + "─" * 50)

    for i, movie in enumerate(results, 1):
        # Create a visual similarity bar using █ characters
        bar_length = int(movie["score"] * 20)  # Scale score (0-1) to bar (0-20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        print(f"  {i}. {movie['title']}")
        print(f"     Similarity: [{bar}] {movie['score']:.2%}")

    print("  " + "─" * 50)
    print()


def main():
    """
    Main function — the entry point of the CLI application.

    Flow:
      1. Display welcome banner
      2. Load and preprocess dataset
      3. Let user choose method (CountVectorizer or TF-IDF)
      4. Build the similarity model
      5. Interactive loop: user types movie names, gets recommendations
    """
    display_banner()

    # --- Step 1: Load & Preprocess the data ---
    print("\n📂 Loading and preprocessing dataset...\n")
    try:
        df = get_prepared_data()
    except FileNotFoundError as e:
        # If the CSV file is missing, show a helpful error and exit
        print(e)
        return

    # --- Step 2: Choose vectorization method ---
    method = choose_method()

    # --- Step 3: Build the recommendation model ---
    print(f"\n🔧 Building recommendation model...")
    recommender = MovieRecommender(df)
    recommender.build_model(method=method)
    print(f"🎉 Model ready! {recommender.get_movie_count()} movies loaded.\n")

    # --- Step 4: Interactive recommendation loop ---
    print("─" * 60)
    print("  Type a movie name to get recommendations.")
    print("  Type 'quit' or 'exit' to stop.")
    print("─" * 60)

    while True:
        # Get input from the user
        user_input = input("\n🎬 Enter a movie name: ").strip()

        # Check if the user wants to quit
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Thanks for using the Movie Recommendation System! Goodbye.\n")
            break

        # Skip empty input
        if not user_input:
            print("  ⚠️  Please type a movie name.")
            continue

        # Get recommendations
        results = recommender.recommend(user_input)

        # Display the results
        display_recommendations(user_input, results)


# ============================================================
# This is the standard Python idiom to run the main function
# when the script is executed directly (not imported)
# ============================================================
if __name__ == "__main__":
    main()
