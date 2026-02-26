import nltk
from nltk.stem import PorterStemmer
import numpy as np

# Ensure proper NLTK data is downloaded once, before initial run
# In production it should be done separately
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    A token can be a word or punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stemming = finding the root form of the word.
    Example:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1.0

    return bag
