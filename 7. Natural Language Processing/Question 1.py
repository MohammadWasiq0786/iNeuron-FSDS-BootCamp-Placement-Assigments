"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Natural Language Processing Assignment 1
"""

"""
Q-1. Take any YouTube videos link and your task is to extract the comments from that videos and store it in a csv file 
and then you need define what is most demanding topic in that videos comment section
"""

# Ans:

import requests
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import models
from gensim.corpora import Dictionary

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to fetch comments from a YouTube video using the YouTube Data API
def fetch_youtube_comments(video_id, api_key):
    base_url = 'https://www.googleapis.com/youtube/v3/commentThreads'
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': 100,
        'key': api_key
    }

    comments = []
    next_page_token = None

    while True:
        if next_page_token:
            params['pageToken'] = next_page_token

        response = requests.get(base_url, params=params)
        data = response.json()

        for item in data['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
            comments.append(comment)

        if 'nextPageToken' in data:
            next_page_token = data['nextPageToken']
        else:
            break

    return comments

# Function to preprocess the comments
def preprocess_comments(comments):
    processed_comments = []

    for comment in comments:
        # Convert to lowercase
        comment = comment.lower()

        # Remove special characters, URLs, and non-alphanumeric characters
        comment = re.sub(r"[^a-zA-Z0-9\s]", "", comment)
        comment = re.sub(r"http\S+|www\S+", "", comment)

        # Tokenize comment into words
        words = word_tokenize(comment)

        # Remove stop words and perform stemming
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words if word not in stop_words]

        # Add processed comment if it contains terms
        if len(words) > 0:
            processed_comments.append(words)

    return processed_comments

# Function to calculate the most demanding topic based on comment count
def find_demanding_topic(comments):
    # Create a dictionary from the comments
    dictionary = Dictionary(comments)

    # Create a corpus (bag of words) representation
    corpus = [dictionary.doc2bow(comment) for comment in comments]

    # Perform LDA topic modeling
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

    # Calculate comment count for each topic
    topic_comment_counts = {topic: 0 for topic in range(lda_model.num_topics)}
    for comment in comments:
        topic = lda_model.get_document_topics(dictionary.doc2bow(comment), minimum_probability=0.2)
        dominant_topic = max(topic, key=lambda x: x[1])[0]
        topic_comment_counts[dominant_topic] += 1

    # Find the most demanding topic based on comment count
    most_demanding_topic = max(topic_comment_counts, key=topic_comment_counts.get)

    return most_demanding_topic

# Function to save comments to a CSV file
def save_comments_to_csv(comments, csv_file):
    df = pd.DataFrame({'Comment': comments})
    df.to_csv(csv_file, index=False)
    print('Comments saved to', csv_file)

# Specify the YouTube video ID and API key
video_id = 'l37XiBGV3fE'
api_key = 'AIzaSyBX-766tW46gpkEjuzO65VfwT9JMUfPeE0'

# Fetch comments from the YouTube video
comments = fetch_youtube_comments(video_id, api_key)

# Check if comments are available
if len(comments) > 0:
    # Specify the CSV file path for comments
    comments_csv_file = 'comments.csv'

    # Save comments to a CSV file
    save_comments_to_csv(comments, comments_csv_file)

    # Preprocess the comments
    preprocessed_comments = preprocess_comments(comments)

    # Perform demanding topic analysis if preprocessed comments are available
    if len(preprocessed_comments) > 0:
        # Find the most demanding topic based on comment count
        most_demanding_topic = find_demanding_topic(preprocessed_comments)

        print("Most Demanding Topic (based on comment count):", most_demanding_topic)
else:
    print("No comments available.")
