"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Natural Language Processing Assignment 4
"""

"""
Q-4. Take any text file and now your task is to Text Summarization without using hugging transformer library
"""

# Ans:

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict

# Read the text file into a Python variable.
text = open("data.txt", "r").read()

# Tokenize the text into sentences.
sentences = sent_tokenize(text)

# Tokenize the sentences into words.
words = word_tokenize(text)

# Remove stop words.
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word.casefold() not in stop_words]

# Calculate word frequencies.
word_frequencies = defaultdict(int)
for word in filtered_words:
    word_frequencies[word] += 1

# Calculate sentence scores based on word frequencies.
sentence_scores = defaultdict(int)
for sentence in sentences:
    for word in word_tokenize(sentence):
        if word.casefold() in word_frequencies:
            sentence_scores[sentence] += word_frequencies[word.casefold()]

# Sort sentences based on their scores in descending order.
sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

# Extract the top sentences to form a summary.
num_sentences = min(3, len(sorted_sentences))  # You can adjust the number of sentences in the summary.
top_sentences = sorted_sentences[:num_sentences]

# Join the extracted sentences together to form the summary.
summary = " ".join(top_sentences)

print(summary)
