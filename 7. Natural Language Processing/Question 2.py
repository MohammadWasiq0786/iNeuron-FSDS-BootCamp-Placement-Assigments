"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Natural Language Processing Assignment 2
"""

"""
Q-2. Take any pdf and your task is to extract the text from that pdf and store it in a .csv file and then you need to find the most repeated word in that pdf.
"""

# Ans:

import PyPDF2
import pandas as pd
import collections
import spacy

# Load the English language model in spaCy
nlp = spacy.load('en_core_web_sm')

# Open the PDF file
pdf_file = open("data.pdf", "rb")

# Create a PDF reader object
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Create a list to store the extracted text
text_list = []

# Iterate over the pages in the PDF file
for page_num in range(len(pdf_reader.pages)):
    # Extract the text from the current page
    page_text = pdf_reader.pages[page_num].extract_text()
    
    # Add the extracted text to the list
    text_list.append(page_text)

# Close the PDF file
pdf_file.close()

# Create a Pandas DataFrame to store the text
df = pd.DataFrame({"Text": text_list})

# Save the DataFrame to a CSV file
df.to_csv("data.csv", index=False)

# Read the CSV file
df = pd.read_csv("data.csv")

# Concatenate all the text from the CSV into a single string
all_text = ' '.join(df["Text"])

# Tokenize the text using spaCy
doc = nlp(all_text)

# Extract the words from the text
words = [token.text.lower() for token in doc if token.is_alpha]

# Create a Counter object to count the occurrences of each word
word_counter = collections.Counter(words)

# Get the most repeated word
most_repeated_word = word_counter.most_common(1)[0][0]

# Print the most repeated word
print("Most repeated word:", most_repeated_word)
