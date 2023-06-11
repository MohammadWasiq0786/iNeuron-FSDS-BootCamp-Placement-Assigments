"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Python Assignment 2
"""

"""
Question 2: -

Consider a string to be valid if all characters of the string appear the same number of times. It is also valid if he can remove just one character at the index in the string, and the remaining characters will occur the same number of times. Given a string, determine if it is valid. If so, return YES , otherwise return NO .

Note - You have to write at least 2 additional test cases in which your program will run successfully and provide an explanation for the same.

Example input 1 - s = “abc”. This is a valid string because frequencies are { “a”: 1, “b”: 1, “c”: 1 }

Example output 1- YES

Example input 2 - s “abcc”. This string is not valid as we can remove only 1 occurrence of “c”. That leaves character frequencies of { “a”: 1, “b”: 1 , “c”: 2 }

Example output 2 - NO
"""

# Ans:

def is_valid(s):
    freq = {}
    for char in s:
        if char in freq:
            freq[char] += 1
        else:
            freq[char] = 1
    values = list(freq.values())
    values.sort()
    unique_values = set(values)
    
    if len(unique_values) == 1:
        return "YES"
    elif len(unique_values) == 2:
        if values.count(values[0]) == 1 and values[0] == 1:
            return "YES"
        elif values.count(values[-1]) == 1 and values[-2] == values[-1] - 1:
            return "NO"  
    
    return "NO"

# Example input 1

s = "abc"
print(is_valid(s))  

# Example output 1 - YES

# Explaination: This is a valid string because frequencies are { “a”: 1, “b”: 1, “c”: 1 }

# Example input 2

s = "abcc"
print(is_valid(s))  

# Example output 2 - NO

# Explaination: This string is not valid as we can remove only 1 occurrence of “c”. That leaves character frequencies of { “a”: 1, “b”: 1 , “c”: 2 }

# Additional Test Cases

# input 3
s = "aabbbcc"
print(is_valid(s))  

# Output: NO

# Explaination: In this case, the string "aabbbcc" has three distinct characters: 'a', 'b', and 'c'. The frequencies of these characters are [2, 3, 2], which are not all equal and cannot be rearranged to form a valid string. No matter how we rearrange the characters, we cannot make the frequencies equal. Therefore, the output is "NO".

# input 4

s = "aabbcc"
print(is_valid(s))  

# Output: YES

# Explaination: In this case, the string "aabbcc" has three distinct characters: 'a', 'b', and 'c'. The frequencies of these characters are [2, 2, 2], which are all equal. Therefore, the output is "YES" since the string is already valid.
