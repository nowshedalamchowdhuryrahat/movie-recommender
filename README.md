# 🎬 Movie Recommendation System

A **content-based Movie Recommendation System** built with Python that suggests similar movies based on their descriptions and genres. Uses Natural Language Processing (NLP) techniques — CountVectorizer and TF-IDF — combined with cosine similarity to find movies that are most alike.

> Built as a portfolio project demonstrating Machine Learning, NLP, and Web Development skills.

---

## ✨ Features

- **Content-Based Filtering** — Recommends movies based on plot descriptions and genres
- **Two Vectorization Methods** — CountVectorizer (simple) and TF-IDF (advanced)
- **Cosine Similarity** — Measures how similar two movies are mathematically
- **Partial Matching** — Type "aveng" and get suggestions for "The Avengers"
- **Case-Insensitive Search** — "avatar", "Avatar", "AVATAR" all work
- **Similarity Scores** — Shows how closely each recommendation matches
- **Flask Web App** — Beautiful dark-themed web interface for searching movies
- **CLI Version** — Terminal-based interface with visual similarity bars

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Main programming language |
| pandas | Data loading and manipulation |
| scikit-learn | CountVectorizer, TF-IDF, cosine similarity |
| Flask | Web application framework |
| HTML/CSS | Web interface with modern dark theme |

---

## 📁 Project Structure

```
movie-recommendation-system/
├── data/
│   └── tmdb_5000_movies.csv    ← Download from Kaggle (see below)
├── templates/
│   └── index.html              ← Web app HTML template
├── static/
│   └── style.css               ← Web app styling
├── data_loader.py              ← Data loading & preprocessing
├── recommender.py              ← Recommendation engine (core ML logic)
├── main.py                     ← Command-line interface
├── app.py                      ← Flask web application
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
└── .gitignore                  ← Git ignore rules
```

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

1. Go to [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
2. Download `tmdb_5000_movies.csv`
3. Place it in the `data/` folder

---

## 💻 Usage

### Option A: Command-Line Interface (CLI)

```bash
python main.py
```

You'll be prompted to:
1. Choose a vectorization method (CountVectorizer or TF-IDF)
2. Type a movie name to get recommendations
3. Type `quit` to exit

### Option B: Flask Web App

```bash
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

---

## 🔬 How It Works

### 1. Data Preprocessing
- Load the TMDB dataset (5000 movies)
- Extract genre names from JSON format
- Combine `overview` + `genres` into a single `tags` column
- Convert to lowercase for consistency

### 2. Vectorization
- **CountVectorizer**: Counts word occurrences to create numerical vectors
- **TF-IDF**: Weighs words by importance — rare words get higher scores

### 3. Cosine Similarity
- Computes the angle between movie vectors in high-dimensional space
- Score of 1.0 = identical, 0.0 = completely different

### 4. Recommendation
- Given a movie, find the top 5 movies with the highest similarity scores

---

## 📊 Example Output

```
🎯 Top 5 movies similar to 'The Dark Knight':
──────────────────────────────────────────────────
1. The Dark Knight Rises
   Similarity: [████████████████████] 100.00%
2. Batman Begins
   Similarity: [████████████████░░░░] 80.25%
3. Batman
   Similarity: [██████████░░░░░░░░░░] 52.14%
4. Batman Returns
   Similarity: [█████████░░░░░░░░░░░] 47.83%
5. Batman Forever
   Similarity: [████████░░░░░░░░░░░░] 41.67%
```

---

## 📝 What I Learned

- Content-based filtering using NLP techniques
- Text vectorization (CountVectorizer vs TF-IDF)
- Cosine similarity for measuring text similarity
- Building web apps with Flask
- Data preprocessing with pandas
- Working with real-world messy datasets

---

## 🔮 Future Improvements

- [ ] Add movie posters using TMDB API
- [ ] Implement collaborative filtering for hybrid recommendations
- [ ] Deploy to Heroku or Render
- [ ] Add user rating system
- [ ] Use Word2Vec or BERT for better text understanding

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Rahat** — Computer Science Student

Built as a portfolio project for graduate school application (KAUST).
