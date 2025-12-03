import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

# Page configuration
st.set_page_config(
    page_title="MAX",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: #0a0a0a;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(255, 153, 0, 0.2), rgba(19, 136, 8, 0.2));
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 153, 0, 0.3);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(255, 153, 0, 0.3);
    }
    
    .movie-card {
        background: #1a1a1a;
        border-radius: 15px;
        padding: 0;
        margin: 1rem 0;
        border: 1px solid #333;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(255, 153, 0, 0.5);
        border-color: #ff9900;
    }
    
    .movie-poster {
        width: 100%;
        height: 400px;
        min-height: 400px;
        max-height: 400px;
        object-fit: cover;
        object-position: center top;
        border-radius: 15px 15px 0 0;
        display: block;
    }
    .movie-info {
        padding: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ff9900 0%, #138808 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(255, 153, 0, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #ff9900 0%, #138808 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px 0 rgba(255, 153, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px 0 rgba(255, 153, 0, 0.7);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #ff9900 0%, #138808 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 12px 40px 0 rgba(255, 153, 0, 0.5);
        margin: 2rem 0;
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 12px 40px 0 rgba(255, 153, 0, 0.5); }
        50% { box-shadow: 0 12px 60px 0 rgba(255, 153, 0, 0.8); }
    }
    
    .math-box {
        background: #1a1a1a;
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        border: 2px solid #333;
        margin: 1rem 0;
    }
    
    .step-box {
        background: #1a1a1a;
        border-left: 4px solid #ff9900;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    h1, h2, h3, h4, p, li, span, div, label {
        color: white !important;
    }
    
    .stProgress > div > div > div {
        background-color: #ff9900;
    }
</style>
""", unsafe_allow_html=True)

# Indian movies dataset with real poster URLs
indian_movies_data = {
    'movie_id': list(range(1, 31)),
    'title': [
        '3 Idiots', 'Dangal', 'PK', 'Baahubali 2', 'Bajrangi Bhaijaan',
        'Secret Superstar', 'Baahubali: The Beginning', 'Sultan', 'Sanju', 'Padmaavat',
        'Tiger Zinda Hai', 'Dilwale Dulhania Le Jayenge', 'Kabir Singh', 'War', 'Chennai Express',
        'Dhoom 3', 'Krrish 3', 'Golmaal Again', 'Simmba', 'Kick',
        'Ek Tha Tiger', 'Bang Bang', 'Sholay', 'Lagaan', 'Andhadhun',
        'Article 15', 'URI: The Surgical Strike', 'Gully Boy', 'Chhichhore', 'Mission Mangal'
    ],
    'year': [
        2009, 2016, 2014, 2017, 2015,
        2017, 2015, 2016, 2018, 2018,
        2017, 1995, 2019, 2019, 2013,
        2013, 2013, 2017, 2018, 2014,
        2012, 2014, 1975, 2001, 2018,
        2019, 2019, 2019, 2019, 2019
    ],
    'genre': [
        'Comedy, Drama', 'Biography, Drama, Sport', 'Comedy, Drama, Sci-Fi', 'Action, Drama', 'Action, Adventure, Comedy',
        'Drama, Music', 'Action, Adventure, Fantasy', 'Action, Drama, Sport', 'Biography, Comedy, Drama', 'Drama, History, Romance',
        'Action, Adventure, Thriller', 'Drama, Romance', 'Drama, Romance', 'Action, Thriller', 'Action, Comedy, Romance',
        'Action, Thriller', 'Action, Sci-Fi', 'Action, Comedy', 'Action, Comedy, Crime', 'Action, Thriller',
        'Action, Romance, Thriller', 'Action, Comedy, Romance', 'Action, Adventure, Thriller', 'Drama, Musical, Sport', 'Crime, Mystery, Thriller',
        'Crime, Drama, Mystery', 'Action, Drama, War', 'Drama, Music, Romance', 'Comedy, Drama', 'Biography, Drama, History'
    ],
    'language': [
        'Hindi', 'Hindi', 'Hindi', 'Telugu', 'Hindi',
        'Hindi', 'Telugu', 'Hindi', 'Hindi', 'Hindi',
        'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi',
        'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi',
        'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi',
        'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi'
    ],
    'poster_url': [
        './3idiots.jpg',  # 3 Idiots
        './dangal.jpg',  # Dangal
        './pk.jpg',  # PK
        './b2.jpg',  # Baahubali 2
        './bb.jpg',  # Bajrangi Bhaijaan
        './Secret Superstar.jpeg',# Secret Superstar
        './Baahubali Beginning.jpg',
        './Sultan.jpg',  # Sultan
        './Sanju.jpg',  # Sanju
        './Padmaavat.jpeg',  # Padmaavat
        './Tiger Zinda Hai.jpg',  # Tiger Zinda Hai
        './DDLJ.jpeg',  # DDLJ
        './Kabir Singh.jpg',  # Kabir Singh
        './War.jpg',  # war
        './Chennai Express.jpg',  # Chennai Express
        './Dhoom 3.jpg',  # Dhoom 3
        './Krrish 3.jpg',  # Krrish 3
        './Golmaal Again.jpeg',  # Golmaal Again
        './Simmba.jpeg',  # Simmba
        './Kick.jpg',  # Kick
        './Ek Tha Tiger.jpg',  # Ek Tha Tiger
        './Bang Bang.jpg',  # Bang Bang
        './Sholay.jpg',  # Sholay
        './Lagaan.jpg',  # Lagaan
        './Andhadhun.jpg',  # Andhadhun
        './Article 15.jpg',  # Article 15
        './URI.jpg',  # URI
        './Gully Boy.jpg',  # Gully Boy
        './Chhichhore.jpeg',  # Chhichhore
        './Mission Mangal.jpg'   # Mission Mangal
    ]
}

indian_movies_df = pd.DataFrame(indian_movies_data)

def create_synthetic_ratings(n_users=100):
    """Create synthetic rating matrix for Indian movies"""
    np.random.seed(42)
    n_movies = len(indian_movies_df)
    
    # Create rating matrix with realistic patterns
    ratings = np.zeros((n_users, n_movies))
    
    for user in range(n_users):
        # Each user rates 40-70% of movies
        n_ratings = np.random.randint(int(n_movies * 0.4), int(n_movies * 0.7))
        movies_to_rate = np.random.choice(n_movies, n_ratings, replace=False)
        
        # User has a base rating tendency (between 2.5 and 4.5)
        user_mean = np.random.uniform(2.5, 4.5)
        
        for movie in movies_to_rate:
            # Add some variance around user mean
            rating = user_mean + np.random.normal(0, 0.7)
            # Clip to valid range [1, 5]
            rating = np.clip(rating, 1.0, 5.0)
            ratings[user, movie] = rating
    
    return ratings
    
    for user in range(n_users):
        # Each user rates 40-70% of movies
        n_ratings = np.random.randint(int(n_movies * 0.4), int(n_movies * 0.7))
        movies_to_rate = np.random.choice(n_movies, n_ratings, replace=False)
        
        # User has a base rating tendency
        user_mean = np.random.normal(3.5, 0.5)
        
        for movie in movies_to_rate:
            # Add some variance
            rating = user_mean + np.random.normal(0, 0.8)
            rating = np.clip(rating, 1, 5)
            ratings[user, movie] = rating
    
    return ratings

def create_user_item_matrix(ratings_array):
    """Create user-item rating matrix"""
    return pd.DataFrame(
        ratings_array,
        index=range(1, len(ratings_array) + 1),
        columns=indian_movies_df['movie_id'].tolist()
    )

def perform_svd_and_predict(user_item_matrix, user_ratings_dict, target_movie_id, k=20):
    """Perform SVD and predict rating"""
    
    # Create new user profile
    new_user_id = user_item_matrix.index.max() + 1
    new_user_ratings = pd.Series(0, index=user_item_matrix.columns)
    
    for movie_id, rating in user_ratings_dict.items():
        if movie_id in new_user_ratings.index:
            new_user_ratings[movie_id] = rating
    
    # Add new user to matrix
    extended_matrix = pd.concat([
        user_item_matrix,
        pd.DataFrame([new_user_ratings.values], 
                     columns=user_item_matrix.columns, 
                     index=[new_user_id])
    ])
    
    # Convert to numpy
    R = extended_matrix.values
    
    # Calculate user means
    user_ratings_mean = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1))
    user_ratings_mean[np.isnan(user_ratings_mean)] = 0
    new_user_mean = user_ratings_mean[-1]
    
    # Normalize ratings
    R_normalized = R - user_ratings_mean.reshape(-1, 1)
    R_normalized[R == 0] = 0
    
    # Perform SVD
    k_components = min(k, min(R.shape) - 1)
    U, sigma, Vt = svds(R_normalized, k=k_components)
    sigma_matrix = np.diag(sigma)
    
    # Reconstruct and predict
    predicted_ratings = np.dot(np.dot(U, sigma_matrix), Vt) + user_ratings_mean.reshape(-1, 1)
    
    # Get prediction for target movie
    movie_idx = list(extended_matrix.columns).index(target_movie_id)
    user_idx = -1
    
    predicted_rating = predicted_ratings[user_idx, movie_idx]
    predicted_rating = max(1, min(5, predicted_rating))
    
    # Calculate similar users
    user_vector = R_normalized[user_idx]
    similarities = []
    for i in range(len(R_normalized) - 1):
        if R[i, movie_idx] > 0:
            other_vector = R_normalized[i]
            dot_product = np.dot(user_vector, other_vector)
            norm_product = np.linalg.norm(user_vector) * np.linalg.norm(other_vector)
            if norm_product > 0:
                similarity = dot_product / norm_product
                similarities.append({
                    'user_id': extended_matrix.index[i],
                    'similarity': similarity,
                    'rating': R[i, movie_idx]
                })
    
    similarities.sort(key=lambda x: abs(x['similarity']), reverse=True)
    top_similar_users = similarities[:5]
    
    explanation_data = {
        'predicted_rating': predicted_rating,
        'user_mean': new_user_mean,
        'n_components': k_components,
        'similar_users': top_similar_users
    }
    
    return explanation_data

def select_random_movies(n_movies=10):
    """Select random movies"""
    import random
    selected = indian_movies_df.sample(n=n_movies, random_state=random.randint(1, 10000))
    return selected.reset_index(drop=True)

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 'welcome'
if 'selected_movies' not in st.session_state:
    st.session_state.selected_movies = None
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
if 'target_movie' not in st.session_state:
    st.session_state.target_movie = None
if 'movie_posters' not in st.session_state:
    st.session_state.movie_posters = {}

if 'ratings_matrix' not in st.session_state:
    st.session_state.ratings_matrix = create_synthetic_ratings(100)

user_item_matrix = create_user_item_matrix(st.session_state.ratings_matrix)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 MAX SVD Recommender</h1>
    <p style="font-size: 1.2rem; color: rgba(255,255,255,0.9);">
        Rate Popular Bollywood Movies & Watch SVD Magic! 🇮🇳
    </p>
</div>
""", unsafe_allow_html=True)

# Welcome Stage
if st.session_state.stage == 'welcome':
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="metric-card" style="padding: 3rem;">
            <h2>🎯 How It Works</h2>
            <p style="font-size: 1.1rem; margin: 1.5rem 0;">
                1️⃣ We'll show you 10 popular Bollywood movies<br><br>
                2️⃣ Rate 9 movies you've watched (1-5 stars)<br><br>
                3️⃣ Our SVD Recommendor will predict your rating for the 10th movie<br><br>
                4️⃣ We'll explain the SVD math behind it!<br><br>
                🎬 Blockbusters like 3 Idiots, Dangal, PK & more!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Start the Challenge!", use_container_width=True):
            st.session_state.selected_movies = select_random_movies(n_movies=10)
            st.session_state.target_movie = st.session_state.selected_movies.iloc[-1]
            st.session_state.stage = 'rating'
            st.rerun()

# Rating Stage
elif st.session_state.stage == 'rating':
    st.markdown("## 🎬 Rate These Bollywood Blockbusters")
    st.markdown("⭐ Rate movies you've **actually watched** - We need 9 ratings!")
    
    progress = len(st.session_state.user_ratings) / 9
    st.progress(progress)
    st.markdown(f"**Progress: {len(st.session_state.user_ratings)}/9 movies rated**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display movies to rate (first 9)
    movies_to_rate = st.session_state.selected_movies.iloc[:-1]
    
    for idx in range(0, len(movies_to_rate), 3):
        cols = st.columns(3)
        for col_idx, col in enumerate(cols):
            movie_idx = idx + col_idx
            if movie_idx < len(movies_to_rate):
                movie = movies_to_rate.iloc[movie_idx]
                
                with col:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    
                    # Display poster from URL
                    st.image(movie['poster_url'], use_container_width=True)
                    
                    st.markdown('<div class="movie-info">', unsafe_allow_html=True)
                    st.markdown(f"### {movie['title']}")
                    st.markdown(f"**Year:** {movie['year']}")
                    st.markdown(f"**Genre:** {movie['genre']}")
                    st.markdown(f"**Language:** {movie['language']}")
                    
                    # Check if already rated
                    if movie['movie_id'] in st.session_state.user_ratings:
                        current_rating = st.session_state.user_ratings[movie['movie_id']]
                        st.success(f"✅ You rated this: {current_rating} ⭐")
                    
                    # Rating buttons
                    st.markdown("**Rate this movie:**")
                    rating_cols = st.columns(5)
                    for i, rating_col in enumerate(rating_cols):
                        with rating_col:
                            rating_value = i + 1
                            if movie['movie_id'] in st.session_state.user_ratings:
                                if st.session_state.user_ratings[movie['movie_id']] == rating_value:
                                    st.button(f"⭐", key=f"rate_{movie['movie_id']}_{rating_value}", 
                                            disabled=True, use_container_width=True)
                                else:
                                    if st.button(f"{rating_value}", key=f"rate_{movie['movie_id']}_{rating_value}",
                                               use_container_width=True):
                                        st.session_state.user_ratings[movie['movie_id']] = rating_value
                                        st.rerun()
                            else:
                                if st.button(f"{rating_value}", key=f"rate_{movie['movie_id']}_{rating_value}",
                                           use_container_width=True):
                                    st.session_state.user_ratings[movie['movie_id']] = rating_value
                                    st.rerun()
                    
                    st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if len(st.session_state.user_ratings) >= 9:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Predict My Rating for the 10th Movie!", use_container_width=True):
                st.session_state.stage = 'prediction'
                st.rerun()
    elif len(st.session_state.user_ratings) > 0:
        st.info(f"💡 You've rated {len(st.session_state.user_ratings)}/9 movies. Rate {9 - len(st.session_state.user_ratings)} more!")

# Prediction Stage
elif st.session_state.stage == 'prediction':
    with st.spinner("🤖 AI is analyzing your preferences using SVD..."):
        explanation_data = perform_svd_and_predict(
            user_item_matrix,
            st.session_state.user_ratings,
            st.session_state.target_movie['movie_id']
        )
    
    st.session_state.explanation_data = explanation_data
    st.session_state.stage = 'reveal'
    st.rerun()

# Reveal Stage
elif st.session_state.stage == 'reveal':
    explanation_data = st.session_state.explanation_data
    target_movie = st.session_state.target_movie
    
    # Show the mystery movie
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(target_movie['poster_url'], use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Prediction for</h2>
            <h1 style="font-size: 2.5rem; margin: 1rem 0;">{target_movie['title']}</h1>
            <div style="font-size: 4rem; font-weight: bold; margin: 1rem 0;">
                {explanation_data['predicted_rating']:.1f} ⭐
            </div>
            <p style="font-size: 1.2rem;">
                Year: {target_movie['year']}<br>
                Genre: {target_movie['genre']}<br>
                Language: {target_movie['language']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Reveal actual rating
    st.markdown("### 🎬 What would YOU actually rate this movie?")
    st.markdown("*(Only if you've watched it!)*")
    
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 2])
    
    for i, col in enumerate([col1, col2, col3, col4, col5]):
        with col:
            if st.button(f"{'⭐' * (i+1)}", key=f"actual_{i}", use_container_width=True):
                st.session_state.actual_rating = i + 1
                st.rerun()
    
    with col6:
        if st.button("❌ Haven't Watched It", key="not_watched", use_container_width=True):
            st.info("No worries! Try another round with different movies.")
            if st.button("🔄 Try Again!", use_container_width=True, key="try_again_not_watched"):
                # Reset everything
                st.session_state.stage = 'welcome'
                st.session_state.selected_movies = None
                st.session_state.user_ratings = {}
                st.session_state.target_movie = None
                st.session_state.movie_posters = {}
                for key in ['actual_rating', 'explanation_data', 'not_watched']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    if 'actual_rating' in st.session_state:
        actual_rating = st.session_state.actual_rating
        difference = abs(explanation_data['predicted_rating'] - actual_rating)
        
        if difference < 0.5:
            st.success(f"🎉 Incredible! SVD was spot on! Difference: {difference:.2f} stars")
        elif difference < 1.0:
            st.info(f"👍 Pretty close! Difference: {difference:.2f} stars")
        elif difference < 1.5:
            st.warning(f"🤔 Not bad! Off by {difference:.2f} stars")
        else:
            st.error(f"😅 Oops! Off by {difference:.2f} stars. You have unique taste!")
        
        st.markdown("---")
        
        # Mathematical Explanation
        st.markdown("## 📊 How Did We Predict This?")
        
        st.markdown("""
        <div class="math-box">
            <h3>🧮 The SVD (Singular Value Decomposition) Magic</h3>
            <p style="font-size: 1.1rem;">
                We analyzed patterns from <strong>100 users</strong> rating <strong>30 Bollywood movies</strong> 
                to find hidden "taste dimensions" and predict your rating!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-box">
            <h4>📊 Step 1: Your Rating Profile</h4>
            <p>Your average rating: <strong>{:.2f} stars</strong></p>
            <p>This tells us if you're generally harsh or generous with ratings.</p>
        </div>
        """.format(explanation_data['user_mean']), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-box">
            <h4>🔢 Step 2: SVD Decomposition</h4>
            <p>We broke down the rating matrix into {} hidden "taste factors":</p>
            <p style="font-size: 1.1rem; margin: 1rem 0;"><code>Rating Matrix ≈ U × Σ × Vᵀ</code></p>
            <ul>
                <li><strong>U (Users × Factors)</strong>: Your preferences across taste dimensions</li>
                <li><strong>Σ (Diagonal)</strong>: Importance of each factor</li>
                <li><strong>Vᵀ (Factors × Movies)</strong>: How movies relate to factors</li>
            </ul>
            <p>Factors might represent: comedy vs drama, old vs new, commercial vs content-driven!</p>
        </div>
        """.format(explanation_data['n_components']), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-box">
            <h4>👥 Step 3: Finding Your Movie Twins</h4>
            <p>Users with similar taste who also rated this movie:</p>
        </div>
        """, unsafe_allow_html=True)
        
        if explanation_data['similar_users']:
            similar_df = pd.DataFrame(explanation_data['similar_users'])
            similar_df['similarity'] = similar_df['similarity'].apply(lambda x: f"{x:.3f}")
            similar_df['rating'] = similar_df['rating'].apply(lambda x: f"{x:.0f} ⭐")
            similar_df.columns = ['User ID', 'Similarity Score (-1 to 1)', 'Their Rating']
            st.dataframe(similar_df, use_container_width=True, hide_index=True)
            st.markdown("*Higher similarity score = more similar taste to you*")
        
        st.markdown("""
        <div class="step-box">
            <h4>🎯 Step 4: The Final Prediction</h4>
            <p><strong>Formula:</strong></p>
            <p style="font-size: 1.1rem;"><code>Predicted Rating = Your Average + (Taste Factor Adjustments)</code></p>
            <p>Starting from your baseline of <strong>{:.2f} stars</strong>, we adjusted based on:</p>
            <ul>
                <li>How similar users rated this movie</li>
                <li>Your preferences across the {} hidden factors</li>
                <li>This movie's characteristics in those factors</li>
            </ul>
            <br>
            <p><strong>🎬 Final Prediction: {:.1f} ⭐</strong></p>
            <p><strong>⭐ Your Actual Rating: {} ⭐</strong></p>
            <p><strong>📏 Difference: {:.2f} stars</strong></p>
        </div>
        """.format(
            explanation_data['user_mean'],
            explanation_data['n_components'],
            explanation_data['predicted_rating'],
            actual_rating,
            difference
        ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Try again button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Try Again with Different Movies!", use_container_width=True):
                # Reset everything
                st.session_state.stage = 'welcome'
                st.session_state.selected_movies = None
                st.session_state.user_ratings = {}
                st.session_state.target_movie = None
                st.session_state.movie_posters = {}
                for key in ['actual_rating', 'explanation_data', 'not_watched']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 1rem;">
    <p>🎬 Popular Bollywood Blockbusters from 1975-2019</p>
</div>
""", unsafe_allow_html=True)
