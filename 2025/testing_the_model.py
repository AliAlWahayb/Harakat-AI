import tensorflow as tf
import numpy as np
import re
import unicodedata
from tensorflow.keras import layers, models
import pandas as pd

# Arabic Text Preprocessing Class
class ArabicPreprocessor:
    def __init__(self):
        self.char_regex = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]')
        self.diacritic_regex = re.compile(r'[\u064B-\u065F\u0610-\u061A]')
        self.normalization_form = 'NFD'

    def clean_text(self, text):
        """Clean Arabic text by removing non-Arabic characters and extra spaces."""
        text = unicodedata.normalize(self.normalization_form, text)
        text = ''.join([char for char in text if self.char_regex.match(char)])
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_diacritics(self, text):
        """Remove diacritics from Arabic text."""
        return re.sub(self.diacritic_regex, '', text)

# Loading the saved model
model = tf.keras.models.load_model('best_model.h5')

# Load vocab (same as during training)
def build_vocab_from_existing_model():
    # Load model's char_to_idx and diacritic_to_idx (you should save these vocabularies during training)
    char_to_idx = {'<PAD>': 0, '<UNK>': 1}  # Replace with your actual vocab
    diacritic_to_idx = {(): 0}  # Replace with your actual vocab
    return char_to_idx, diacritic_to_idx

char_to_idx, diacritic_to_idx = build_vocab_from_existing_model()

# Reverse the diacritic vocab to map indices back to diacritics
idx_to_diacritic = {v: k for k, v in diacritic_to_idx.items()}

# Preprocessing text before prediction
preprocessor = ArabicPreprocessor()

def predict_harakat(text):
    # Clean and preprocess the input text
    cleaned_text = preprocessor.clean_text(text)
    sequence = [char_to_idx.get(c, 1) for c in cleaned_text]  # Use <UNK> for unknown chars
    max_len = 150  # Define the max length used during training
    sequence = sequence[:max_len] + [0] * (max_len - len(sequence))  # Padding to max_len
    
    # Make prediction using the model
    prediction = model.predict(np.array([sequence]))
    
    # Debug: Print the predicted indices
    predicted_indices = np.argmax(prediction[0], axis=-1)
    print("Predicted indices:", predicted_indices)

    # Map predicted indices back to diacritics
    diacritics = []
    for i in predicted_indices:
        diacritic = idx_to_diacritic.get(i, ())  # Use empty tuple for unknown diacritics
        diacritics.append(diacritic)
    
    # Debug: Print the diacritics
    print("Diacritics:", diacritics)
    
    # Reconstruct the original text with diacritics
    result = []
    for char, diacritic in zip(cleaned_text, diacritics):
        result.append(char)
        result.extend(diacritic)
    
    return ''.join(result)

# Example usage
test_text = "اللغة العربية جميلة"
predicted_text = predict_harakat(test_text)

print("Input:", test_text)
print("Predicted with Harakat:", predicted_text)
