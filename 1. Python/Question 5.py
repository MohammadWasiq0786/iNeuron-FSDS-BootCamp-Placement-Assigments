"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Python Assignment 5
"""

"""
Question 5 -
Write a program to download the data from the given API link and then extract the following data with
proper formatting
Link - http://api.tvmaze.com/singlesearch/shows?q=westworld&embed=episodes
Note - Write proper code comments wherever needed for the code understanding
Excepted Output Data Attributes -
● id - int url - string
● name - string season
● - int number - int
● type - string airdate -
● date format airtime -
● 12-hour time format
● runtime - float
● average rating - float
● summary - string
● without html tags
● medium image link - string
● Original image link - string
"""

# Ans:

import requests
import json
from bs4 import BeautifulSoup

# API URL
url = "http://api.tvmaze.com/singlesearch/shows?q=westworld&embed=episodes"

# Send a GET request to the API URL and retrieve the data
response = requests.get(url)
data = json.loads(response.text)

# Iterate over each episode and extract the required data
for episode in data['_embedded']['episodes']:
    # Extract attributes
    episode_id = episode['id']
    episode_url = episode['url']
    episode_name = episode['name']
    season = episode['season']
    episode_number = episode['number']
    episode_type = episode['type']
    airdate = episode['airdate']
    airtime = episode['airtime']
    runtime = episode['runtime']
    average_rating = episode['rating']['average']
    
    # Clean up the summary by removing HTML tags using BeautifulSoup
    summary = BeautifulSoup(episode['summary'], 'html.parser').get_text(strip=True)
    
    # Extract image links
    image_medium = episode['image']['medium']
    image_original = episode['image']['original']
    
    # Print the extracted data with proper formatting
    print("Episode ID:", episode_id)
    print("URL:", episode_url)
    print("Name:", episode_name)
    print("Season:", season)
    print("Number:", episode_number)
    print("Type:", episode_type)
    print("Airdate:", airdate)
    print("Airtime:", airtime)
    print("Runtime:", runtime)
    print("Average Rating:", average_rating)
    print("Summary:", summary)
    print("Medium Image Link:", image_medium)
    print("Original Image Link:", image_original)
    print()