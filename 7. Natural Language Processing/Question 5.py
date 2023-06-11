"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Natural Language Processing Assignment 5
"""

'''
Q5. Now you need build your own language detection with the fast Text model by Facebook.
'''

# Ans:


import fasttext

# Load pre-trained language detection model
model_path = 'lid.176.bin'  # Path to the pre-trained model file
model = fasttext.load_model(model_path)

# Function to detect language given a text
def detect_language(text):
    predictions = model.predict(text, k=1)  # Get top predicted language
    predicted_language = predictions[0][0].split('__')[-1]  # Extract language label
    return predicted_language

# Example usage
text = "Hello, how are you?"
predicted_language = detect_language(text)
print(f"Detected language: {predicted_language}")



