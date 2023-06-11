"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Natural Language Processing Assignment 3
"""

"""
Q-3. from question 2, As you got the CSV and now you need perform key word extraction from that csv file and do the Topic modeling.
"""

# Ans:

import csv
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import models
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Read the CSV file
csv_file = 'data.csv'
text_column = 'Text'  # Replace with the column name containing the text data

df = pd.read_csv(csv_file)
texts = df[text_column].tolist()

# Preprocess the data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

processed_texts = [preprocess_text(text) for text in texts]

# Perform keyword extraction using TF-IDF
dictionary = Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]
tfidf_model = TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]

# Extract the top keywords from the corpus
top_keywords = []
for doc in tfidf_corpus:
    sorted_keywords = sorted(doc, key=lambda x: x[1], reverse=True)[:5]  # Extract top 5 keywords from document
    keywords = [dictionary[word_id] for (word_id, _) in sorted_keywords]
    top_keywords.append(keywords)

# Perform topic modeling using LDA
lda_model = LdaModel(tfidf_corpus, num_topics=5, id2word=dictionary)

# Analyze and interpret the results
for i, topic in lda_model.show_topics():
    print(f"Topic #{i+1}: {topic}\n")

print("Top keywords for the document:")
print(top_keywords[0])  # Print top keywords for the document
