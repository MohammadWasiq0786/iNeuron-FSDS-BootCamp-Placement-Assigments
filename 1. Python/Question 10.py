"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Python Assignment 10
"""

"""
Question 10 -

Write a program to count the number of verbs, nouns, pronouns, and adjectives in a given particular phrase or paragraph, and return their respective count as a dictionary.
Note -
1. Write code comments wherever required for code
2. You have to write at least 2 additional test cases in which your program will run successfully and provide an explanation for the same.
"""

# Ans:

import re

def count_pos_tags(text):
    # Define regular expressions for matching different parts of speech
    noun_pattern = re.compile(r'\b[A-Za-z]+[s]?\b', re.IGNORECASE)
    pronoun_pattern = re.compile(r'\b(I|me|my|mine|you|your|yours|he|him|his|she|her|hers|it|its|we|us|our|ours|they|them|their|theirs)\b', re.IGNORECASE)
    verb_pattern = re.compile(r'\b[A-Za-z]+(?:s|ed|ing)?\b', re.IGNORECASE)
    adjective_pattern = re.compile(r'\b[A-Za-z]+\b', re.IGNORECASE)

    # Initialize counts
    noun_count = 0
    pronoun_count = 0
    verb_count = 0
    adjective_count = 0

    # Find matches for each part of speech
    noun_matches = re.findall(noun_pattern, text)
    pronoun_matches = re.findall(pronoun_pattern, text)
    verb_matches = re.findall(verb_pattern, text)
    adjective_matches = re.findall(adjective_pattern, text)

    # Count the matches
    noun_count = len(noun_matches)
    pronoun_count = len(pronoun_matches)
    verb_count = len(verb_matches)
    adjective_count = len(adjective_matches)

    # Create and return the dictionary
    pos_counts = {
        'nouns': noun_count,
        'pronouns': pronoun_count,
        'verbs': verb_count,
        'adjectives': adjective_count
    }
    return pos_counts

# Test case 1
text = "I love to eat pizza."
result = count_pos_tags(text)
print(result)

#Test case 2
text = "The cat chased the mouse. It was quick and agile. The dog barked loudly."
result = count_pos_tags(text)
print(result)


