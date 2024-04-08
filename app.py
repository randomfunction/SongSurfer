import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Function to load data
@st.cache_data
def load_data():
    songs = pd.read_csv('data.csv')
    return songs

# Function to preprocess data
@st.cache_data
def preprocess_data(songs):
    songs = songs[['name', 'valence', 'year', 'acousticness', 'artists', 'danceability', 'duration_ms', 'energy', 'explicit', 'instrumentalness', 'id', 'liveness', 'key', 'loudness', 'mode', 'speechiness', 'tempo']]

    # Create tag column
    songs['tag'] = songs.drop(columns=['id','name']).apply(lambda x: ','.join(x.astype(str)), axis=1)
    songs['tag'] = songs['tag'].astype(str)
    return songs

# Function to compute TF-IDF vectors
@st.cache_data
def compute_tfidf_vectors(songs):
    tfidf_vectorizer = TfidfVectorizer()
    tag_vectors = tfidf_vectorizer.fit_transform(songs['tag'])
    return tag_vectors

# Function to find top similar songs
def find_top_similar_songs(target_song_name, tag_vectors, songs):
    target_song = songs[songs['name'].str.lower() == target_song_name.lower()]
    if target_song.empty:
        st.error("Error: Target song not found.")
        return []

    target_song_index = target_song.index[0]
    target_song_vector = tag_vectors[target_song_index]

    # Calculate cosine similarity excluding the target song
    similarities = cosine_similarity(target_song_vector.reshape(1, -1), tag_vectors)[0]
    similarities[target_song_index] = -1  # Exclude similarity to the target song

    # Get indices of top 10 similar songs
    top_10_indices = np.argsort(similarities)[::-2][:11]
    top_10_song_ids = songs.iloc[top_10_indices]['name'].tolist()
    return top_10_song_ids

# Streamlit app
def main():
    st.title('Song Recommendation System')

    songs = load_data()
    songs = preprocess_data(songs)
    tag_vectors = compute_tfidf_vectors(songs)

    target_song_name = st.text_input('Enter a song name:', 'Danny Boy')
    if st.button('Find Similar Songs'):
        top_similar_songs = find_top_similar_songs(target_song_name.lower(), tag_vectors, songs)
        if top_similar_songs:
            st.write('Top 10 similar songs:')
            for i, song in enumerate(top_similar_songs, 1):
                st.write(f'{i}. {song}')
        else:
            st.write('No similar songs found.')

if __name__ == "__main__":
    main()
