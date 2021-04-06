# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title='Song Recommendation App',
    initial_sidebar_state='expanded'
)

# Load Dataset
@st.cache
def load_data(data):
    df = pd.read_csv(data)
    df.drop(columns ='Unnamed: 0', inplace=True)
    df['genres'] = df['genres'].apply(lambda x: x[1:-1].split(', '))
    for i in df.index:
        df['genres'].loc[i] = list(filter(None, df['genres'][i]))
    return df

@st.cache
def load_data_recommender(data):
    recommender_df = pd.read_csv(data)
    recommender_df.drop(columns ='Unnamed: 0', inplace=True)
    recommender_df['genres'] = recommender_df['genres'].apply(lambda x: x[1:-1].split(', '))
    for i in recommender_df.index:
        recommender_df['genres'].loc[i] = list(filter(None, recommender_df['genres'][i]))
    return recommender_df

# Recommendation System
@st.cache
def song_recommender(data, song, artist, genre_parameter):
    song_and_artist_data = data[(data['artists'] == artist) & (data['song_name'] == song)].sort_values('year')[0:1]
    
    similarity_data = data.copy()
    
    data_values = similarity_data[['acousticness', 'danceability',
       'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'valence']]
    
    similarity_data['similarity_with_song'] =cosine_similarity(data_values, data_values.to_numpy()[song_and_artist_data.index[0],None]).squeeze()
    
    artist_genres = set(*song_and_artist_data['genres'])

    similarity_data['genres'] = similarity_data['genres'].apply(lambda genres: list(set(genres).intersection(artist_genres)))
    
    similarity_lengths = similarity_data['genres'].str.len()
    similarity_data = similarity_data.reindex(similarity_lengths[similarity_lengths >= genre_parameter].sort_values(ascending=False).index)
    
    similarity_data = similarity_data[similarity_data['song_decade'] == song_and_artist_data['song_decade'].values[0]]
 
    similarity_data.rename(columns={'song_name': f'Similar Song to {song}'}, inplace=True)
    
    similarity_data = similarity_data.sort_values(by= 'similarity_with_song', ascending = False)
    
    similarity_data = similarity_data[['artists', f'Similar Song to {song}',
       'song_popularity', 'year', 'genres', 'artist_popularity', 'song_decade', 'similarity_with_song',
       'acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']]
    results = similarity_data[['artists', f'Similar Song to {song}', 'year', 'genres', 'similarity_with_song','song_popularity', 'artist_popularity']]
    return results[1:11]

# If Not Found
@st.cache
def song_title_not_found(term, data):
    result_df = rec_df[rec_df['song_name'].str.contains(term)]
    return result_df


st.title('Song Recommender App')


st.sidebar.title('Menu')
st.sidebar.text('Select a page')
menu = st.sidebar.selectbox(
    'Menu',
    ('Song Recommender', 'About')
)

df = load_data('Datasets/df_cleaned.csv')
rec_df = load_data_recommender('Datasets/final_df.csv')

if menu == 'Song Recommender':
    st.subheader('Enter Your Favorite Song. Get a List of 10 New Songs to Add to Your Playlist.')
    song_title = st.text_input('Song Title')
    song_artist = st.text_input('Artist')
    song_genre = st.number_input('Number of Similar Genres', min_value=1, max_value=5, step=1)
    if st.button('Recommend'):
        if song_title is not None:
            try:
                result = song_recommender(rec_df, song_title, song_artist, song_genre)
                st.write(result)
            except:
                result = 'Not Found'
                st.warning(result)
                st.info('Suggested Options')
                result_df = song_title_not_found(song_title, rec_df)
                st.dataframe(result_df)
elif menu == 'About':
    st.subheader('About this project')
    st.write("""
    
    The data used for this recommender system came from Spotify and I was able to download the datasets from Kaggle.  After cleaning the data, the dataset used for the recommender contains 144,166 songs from 1920 to 2021.  
    
    To build this song recommender system we used Spotify's song metrics to calculate the cosine similarity of each song's metrics.  The cosine similarity will give a similarity score to determine how similar two songs are.  The song metrics used in this recommender system include: acousticness, danceability, energy, instrumentalness, key, liveness, loudness, mode,speechiness, tempo, and valence. 
    
    """)