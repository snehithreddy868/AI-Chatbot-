import json
import random
import numpy as np
import nltk
from nlp_utils import tokenize, stem, bag_of_words

# Pre-download required nltk packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class BasicChatbot:
    def __init__(self, intents_path):
        with open(intents_path, 'r') as f:
            self.intents = json.load(f)
        
        self.all_words = []
        self.tags = []
        self.xy = []

        self._prepare_data()

    def _prepare_data(self):
        """Builds vocabulary and pattern mappings from intents file."""
        for intent in self.intents['intents']:
            tag = intent['tag']
            if tag not in self.tags:
                self.tags.append(tag)
            
            for pattern in intent['patterns']:
                w = tokenize(pattern)
                self.all_words.extend(w)
                self.xy.append((w, tag))
        
        ignore_words = ['?', '!', '.', ',']
        self.all_words = [stem(w) for w in self.all_words if w not in ignore_words]
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))

    def get_response(self, text):
        """Returns the best response by measuring word overlap with patterns."""
        sentence_words = tokenize(text)
        user_bag = bag_of_words(sentence_words, self.all_words)

        best_match_tag = None
        max_overlap = 0

        # Compare user input to all known patterns
        for pattern_words, tag in self.xy:
            pattern_bag = bag_of_words(pattern_words, self.all_words)
            
            # Dot product measures the number of matching words
            overlap = np.dot(user_bag, pattern_bag)

            if overlap > max_overlap:
                max_overlap = overlap
                best_match_tag = tag

        # Filter out random matches with low overlap threshold (optional, keeping it simple as > 0)
        if max_overlap > 0 and best_match_tag:
            return self._get_random_response(best_match_tag)
        
        return "I'm sorry, I don't understand."

    def _get_random_response(self, tag):
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        return "I'm sorry, I don't understand."
