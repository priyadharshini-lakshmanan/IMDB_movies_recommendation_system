import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords

# Page config
st.set_page_config(
    page_title="IMDB Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_and_preprocess_data(file_path):
    """Load CSV and preprocess storylines"""
    df = pd.read_csv(file_path)
    
    # Fix column names
    if 'Movie Name' in df.columns:
        df['Movie_name'] = df['Movie Name']
    if 'storyline' in df.columns.str.lower():
        df['Storyline'] = df[df.columns[df.columns.str.lower() == 'storyline'].tolist()[0]]
    
    # Download stopwords if needed
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(filtered)
    
    df['clean_storyline'] = df['Storyline'].apply(clean_text)
    
    # Build TF-IDF
    corpus = df['clean_storyline'].dropna()
    corpus = corpus[corpus.str.strip() != '']
    
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, vectorizer, tfidf_matrix, similarity_matrix

# Header
st.title("🎬 IMDB Movie Recommender System")

# Load data
try:
    df, vectorizer, tfidf_matrix, similarity_matrix = load_and_preprocess_data("C:\\Users\\ACER\\Desktop\\cleaned_imdb_2024.csv")
    # st.success(f"✅ Loaded {len(df)} movies!")
except FileNotFoundError:
    st.error("❌ **cleaned_imdb_2024.csv** not found in current folder!")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Main tabs
tab1, tab2 = st.tabs(["🔍 Storyline Search", "🎯 Movie Search"])

with tab1:
    st.header("📝 Find Movies by Storyline")
    
    # Input
    storyline = st.text_area(
        "Enter a movie storyline:",
        placeholder="E.g., A young wizard discovers he is famous at magical school...",
        height=120
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        top_n = st.selectbox("Show top:", [5, 10], index=0)
    
    if st.button("🔍 Find Similar Movies", type="primary"):
        if storyline:
            with st.spinner("Finding similar movies..."):
                # Clean input
                stop_words = set(stopwords.words('english'))
                clean_input = re.sub(r'[^a-z\s]', '', storyline.lower())
                words = clean_input.split()
                clean_input = ' '.join([w for w in words if w not in stop_words and len(w) > 2])
                
                # TF-IDF transform
                input_vector = vectorizer.transform([clean_input])
                similarities = cosine_similarity(input_vector, tfidf_matrix)[0]
                
                # Get top matches
                scores_idx = np.argsort(similarities)[::-1][:top_n]
                
                st.success(f"✅ Found {top_n} similar movies!")
                
                # Results
                for i, idx in enumerate(scores_idx):
                    score = similarities[idx]
                    movie_name = df.iloc[idx]['Movie_name']
                    storyline_text = df.iloc[idx]['Storyline']
                    
                    with st.container():
                        st.markdown(f"""
                        
                            <h3>🎥 {i+1}. {movie_name}</h3>
                            <p><strong>Similarity:</strong> {score:.3f}</p>
                                {storyline_text[:200]}...
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a storyline!")

with tab2:
    st.header("🎯 Find Similar Movies")
    
    # Movie dropdown
    movie_options = df['Movie_name'].tolist()
    selected_movie = st.selectbox("Select a movie:", movie_options)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        top_n_movie = st.selectbox("Show top:", [5, 10], index=0, key="top_n_movie")
    
    if st.button("🎯 Get Recommendations", type="primary"):
        # Find movie index
        movie_idx = df[df['Movie_name'] == selected_movie].index[0]
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n_movie+1]
        
        st.success(f"✅ Top {top_n_movie} movies like **{selected_movie}**!")
        
        # Results
        for i, (idx, score) in enumerate(sim_scores):
            movie_name = df.iloc[idx]['Movie_name']
            storyline_text = df.iloc[idx]['Storyline']
            
            with st.container():
                st.markdown(f"""
                    <h3>🎥 {i+1}. {movie_name}</h3>
                    <p><strong>Similarity:</strong> {score:.3f}</p>
                        {storyline_text[:200]}...
                    </p>
                </div>
                """, unsafe_allow_html=True)


