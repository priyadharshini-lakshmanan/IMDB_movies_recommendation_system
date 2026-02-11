# 🎬 IMDb Movie Recommendation System

A movie recommendation system that suggests similar movies based on storyline analysis using NLP and Machine Learning.

## 🎯 Overview

This project scrapes 2024 movie data from IMDb and uses TF-IDF vectorization and Cosine Similarity to recommend movies with similar storylines.

## ✨ Features

- Web scraping of 250 movies from IMDb 2024
- NLP-based storyline analysis using TF-IDF
- Cosine Similarity for movie matching
- Interactive Streamlit web interface
- Top 5 movie recommendations

## 🛠️ Tech Stack

- **Python 3.11+**
- **Selenium** - Web scraping
- **Pandas** - Data processing
- **Scikit-learn** - TF-IDF & Cosine Similarity
- **Streamlit** - Web interface
- **NLTK** - Text preprocessing

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/imdb-movie-recommendation.git
cd imdb-movie-recommendation

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Scrape Data
```bash
jupyter notebook imdb.ipynb
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```

### 3. Get Recommendations
- Enter a movie storyline
- Click "Get Recommendations"
- View top 5 similar movies

## 📊 How It Works

1. **Scrape** movie data from IMDb using Selenium
2. **Preprocess** storylines (cleaning, tokenization)
3. **Vectorize** text using TF-IDF
4. **Calculate** similarity using Cosine Similarity
5. **Recommend** top 5 most similar movies

## 📁 Project Structure

```
├── imdb.ipynb              # Data scraping & analysis
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── cleaned_imdb_2024.csv   # Processed data
└── README.md              # Documentation
```

## 📝 Requirements

```
pandas
numpy
selenium
scikit-learn
streamlit
nltk
beautifulsoup4
```

## 🎓 What I Learned

- Web scraping with Selenium
- NLP text preprocessing
- TF-IDF vectorization
- Cosine Similarity algorithm
- Building interactive web apps with Streamlit

## 🔮 Future Improvements

- Add genre filtering
- Include movie ratings
- Deploy to cloud
- Add more visualization
