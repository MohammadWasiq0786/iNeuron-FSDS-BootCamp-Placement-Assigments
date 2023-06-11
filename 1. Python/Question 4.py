"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Python Assignment 4
"""

"""
Question 4 -

Write a program to download the data from the link given below and then read the data and convert the into
the proper structure and return it as a CSV file.

Link - https://data.nasa.gov/resource/y77d-th95.json

Note - Write code comments wherever needed for code understanding.

Excepted Output Data Attributes

● Name of Earth Meteorite - string id - ID of Earth

● Meteorite - int nametype - string recclass - string

● mass - Mass of Earth Meteorite - float year - Year at which Earth

● Meteorite was hit - datetime format reclat - float recclong - float

● point coordinates - list of int
"""

# Ans:

import requests
import json
import csv

def download_data(url):
    response = requests.get(url)
    data = response.json()
    return data

def convert_to_csv(data, filename):
    # Extracting the expected output data attributes from the JSON data
    attributes = [
        "name",
        "id",
        "nametype",
        "recclass",
        "mass",
        "year",
        "reclat",
        "reclong",
        "geolocation.coordinates"
    ]

    # Opening the CSV file and writing the header
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(attributes)

        # Writing each record as a row in the CSV file
        for item in data:
            row = []
            for attribute in attributes:
                # Handling nested attributes
                if '.' in attribute:
                    nested_attributes = attribute.split('.')
                    nested_value = item.get(nested_attributes[0], {})
                    for nested_attribute in nested_attributes[1:]:
                        nested_value = nested_value.get(nested_attribute, '')
                    row.append(nested_value)
                else:
                    row.append(item.get(attribute, ''))
            writer.writerow(row)

    print("CSV file generated successfully.")

# URL of the JSON data
url = "https://data.nasa.gov/resource/y77d-th95.json"

# Download the data from the URL
data = download_data(url)

# Convert and save the data as a CSV file
convert_to_csv(data, "Output.csv")
