"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Natural Language Processing Assignment 6
"""

'''
Q6. Generate research papers titles using Bert model and containerize the application and push to public docker hub.
'''

# Ans:

import tensorflow as tf
import bert

def generate_title(text):
  """Generates a title for the given text.

  Args:
    text: The text to generate a title for.

  Returns:
    A title for the given text.
  """

  # 1. Tokenize the text.
  tokenizer = bert.BertTokenizer(vocab_file='bert_model/vocab.txt')
  tokens = tokenizer.tokenize(text)

  # 2. Convert the tokens to a sequence of integers.
  encoded_tokens = tokenizer.convert_tokens_to_ids(tokens)

  # 3. Create a BERT model.
  model = tf.keras.models.load_model('bert_model/model.h5')

  # 4. Generate a title.
  title = model.predict(encoded_tokens)[0]

  # 5. Return the title.
  return title

def main():
  # 1. Get the text of the research paper.
  text = """
  The paper presents a new method for generating research papers titles. The method is based on a BERT model. The BERT model is trained on a large corpus of research papers titles. The method is evaluated on a test set of research papers titles. The results show that the method is able to generate titles that are similar to the titles of human-written research papers.
  """

  # 2. Generate a title.
  title = generate_title(text)

  # 3. Print the title.
  print(title)

if __name__ == '__main__':
  main()



