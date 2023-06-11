"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Natural Language Processing Assignment 7
"""

'''
Q7. Now you need to build your own chatbot using the seq2seq model of Amazon website by scrape the website 
and containerize the application and push to public docker hub.
'''

# Ans:

import requests
import bs4
import tensorflow as tf

# Scrape the Amazon website
response = requests.get('https://www.amazon.com/')
soup = bs4.BeautifulSoup(response.content, 'html.parser')

# Find all the product listings
product_listings = soup.find_all('div', class_='s-product-listing')

# Create a list of product titles
product_titles = []
for product_listing in product_listings:
    product_title = product_listing.find('a', class_='a-link-normal a-color-base a-text-normal').text
    product_titles.append(product_title)

# Create a seq2seq model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(product_titles), 128),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(product_titles), activation='softmax')
])

# Train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(product_titles, product_titles, epochs=10)

# Save the model
model.save('amazon_chatbot.h5')


