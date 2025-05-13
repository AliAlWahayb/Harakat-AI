# Step 1: Import Required Libraries
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from sklearn.model_selection import train_test_split
import unicodedata
import re
import matplotlib.pyplot as plt
# Step 2: Data Loading and Preprocessing
class ArabicPreprocessor:
    def __init__(self):
        self.char_regex = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]')
        self.diacritic_regex = re.compile(r'[\u064B-\u065F\u0610-\u061A]')
        self.normalization_form = 'NFD'

    def clean_text(self, text):
        """Clean Arabic text by removing non-Arabic characters and extra spaces."""
        # Normalize the text
        text = unicodedata.normalize(self.normalization_form, text)
        # Keep only Arabic characters and spaces
        text = ''.join([char for char in text if self.char_regex.match(char)])
        # Normalize spaces and remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_diacritics(self, text):
        """Extract and sort diacritics to handle order variations."""
        normalized = unicodedata.normalize(self.normalization_form, text)
        diacritics = []
        current_diacritics = []
        
        for c in normalized:
            if unicodedata.category(c).startswith('Mn'):
                current_diacritics.append(c)
            else:
                if diacritics:
                    diacritics[-1] = tuple(sorted(current_diacritics, key=lambda x: unicodedata.name(x)))
                diacritics.append(tuple())
                current_diacritics = []
        
        if diacritics:
            diacritics[-1] = tuple(sorted(current_diacritics, key=lambda x: unicodedata.name(x)))
        else:
            diacritics.append(tuple(current_diacritics))
            
        return diacritics

    def remove_diacritics(self, text):
        """Remove diacritics from Arabic text."""
        # Using regular expression to remove diacritic marks
        return re.sub(self.diacritic_regex, '', text)



# Step 3: Data Preparation
def load_data(folder_path):
    preprocessor = ArabicPreprocessor()
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(all_files)} files in {folder_path}")
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    pairs = []
    
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # Check if 'text_with_harakat' column exists
            if 'text_with_harakat' not in df.columns:
                print(f"Warning: 'text_with_harakat' column missing in {file_path}")
                continue
            
            for _, row in df.iterrows():
                original_text = row['text_with_harakat']
                if not isinstance(original_text, str):
                    print(f"Warning: Invalid data type in {file_path}, row {_}, skipping.")
                    continue
                
                clean_output = preprocessor.clean_text(original_text)
                clean_input = preprocessor.remove_diacritics(clean_output)
                diacritics = preprocessor.extract_diacritics(clean_output)
                
                # Critical length validation
                if len(clean_input) != len(diacritics):
                    print(f"Warning: Length mismatch in {file_path}, row {_}, skipping.")
                    continue
                    
                pairs.append((clean_input, diacritics))
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue
            
    return pairs

# Step 4: Vocabulary Creation
def build_vocab(pairs):
    char_vocab = set()
    diacritic_vocab = set()
    
    for input_text, diacritics in pairs:
        char_vocab.update(input_text)
        diacritic_vocab.update(diacritics)
    
    # Add special tokens
    char_to_idx = {'<PAD>': 0, '<UNK>': 1}
    char_to_idx.update({char: i+2 for i, char in enumerate(char_vocab)})
    
    # Ensure () is always index 0 for no diacritics
    diacritic_to_idx = {(): 0}
    diacritic_to_idx.update({d: i+1 for i, d in enumerate([d for d in diacritic_vocab if d])})
    
    return char_to_idx, diacritic_to_idx

def verify_vocab(pairs, diacritic_to_idx):
    valid_diacritics = set(diacritic_to_idx.keys())
    for _, diacritics in pairs:
        for d in diacritics:
            if d not in valid_diacritics:
                print(f"Found unseen diacritic combination: {d}")
                return False
    return True
# Step 5: Sequence Preparation
def prepare_sequences(pairs, char_to_idx, diacritic_to_idx, max_len):
    input_seqs = []
    output_seqs = []
    
    valid_diacritics = set(diacritic_to_idx.keys())
    
    for input_text, diacritics in pairs:
        # Validate all diacritics are in vocabulary
        if any(d not in valid_diacritics for d in diacritics):
            continue
            
        input_seq = [char_to_idx.get(c, 1) for c in input_text]
        output_seq = [diacritic_to_idx[d] for d in diacritics]
        
        if len(input_seq) > max_len or len(output_seq) != len(input_seq):
            continue
            
        # Padding
        pad_length = max_len - len(input_seq)
        input_seq += [0] * pad_length
        output_seq += [0] * pad_length
        
        input_seqs.append(input_seq)
        output_seqs.append(output_seq)
    
    return np.array(input_seqs, dtype='int32'), np.array(output_seqs, dtype='int32')
# Step 6: Model Architecture
def build_model(input_vocab_size, output_vocab_size, max_len):
    model = models.Sequential([
        layers.Embedding(input_vocab_size, 256, input_length=max_len),
        layers.Bidirectional(layers.LSTM(512, return_sequences=True)),
        layers.Dropout(0.4),
        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        layers.Dropout(0.4),
        layers.TimeDistributed(layers.Dense(128, activation='relu')),
        layers.TimeDistributed(layers.Dense(output_vocab_size, activation='softmax'))
    ])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

# Step 7: Training Configuration
class HarakatTrainer:
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.callbacks = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
            callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
    
    def train(self, epochs=50, batch_size=128):
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks
        )
        return history

# Step 8: Visualization
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
# Step 9: Inference Engine
class HarakatPredictor:
    def __init__(self, model, char_to_idx, diacritic_to_idx, max_len):
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_diacritic = {v: k for k, v in diacritic_to_idx.items()}
        self.max_len = max_len
        
    def predict(self, text):
        cleaned = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        sequence = [self.char_to_idx.get(c, 1) for c in cleaned]
        sequence = sequence[:self.max_len] + [0] * (self.max_len - len(sequence))
        
        prediction = self.model.predict(np.array([sequence]))
        diacritics = [self.idx_to_diacritic[i] for i in np.argmax(prediction[0], axis=-1)]
        
        result = []
        for char, diacritic in zip(cleaned, diacritics):
            result.append(char)
            result.extend(diacritic)
        return ''.join(result)

# Main Execution Flow
if __name__ == '__main__':
    # Load and prepare data
    data_folder = "data"
    pairs = load_data(data_folder)
    if not pairs:
        print("No data loaded. Please check the data folder and file format.")
    else:
        print("Input:", pairs[0][0])
        print("Diacritics:", pairs[0][1])

    print(f"Initial pairs loaded: {len(pairs)}")
    
    # if not pairs:
    #     raise ValueError("No valid data found in the specified folder")
        
    # Inspect sample data
    print("\nSample input-output pair:")
    print("Input:", pairs[0][0])
    print("Diacritics:", pairs[0][1])
    
    # Build vocabulary
    char_to_idx, diacritic_to_idx = build_vocab(pairs)
    print(f"\nCharacter vocabulary size: {len(char_to_idx)}")
    print(f"Diacritic vocabulary size: {len(diacritic_to_idx)}")
    
    # Set max sequence length based on data
    lengths = [len(p[0]) for p in pairs]
    max_len = min(200, max(lengths)) if lengths else 150
    print(f"\nMax sequence length: {max_len}")
    print(f"95th percentile length: {np.percentile(lengths, 95) if lengths else 0}")
    
    # Prepare sequences with validation
    X, y = prepare_sequences(pairs, char_to_idx, diacritic_to_idx, max_len)
    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    
    if X.shape[0] == 0:
        raise ValueError("No valid sequences after preprocessing. Check:")
        "- Text normalization and diacritic extraction"
        "- Vocabulary coverage"
        "- Sequence length filtering"
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model with verified output size
    model = build_model(len(char_to_idx), len(diacritic_to_idx), max_len)
    trainer = HarakatTrainer(model, X_train, y_train, X_val, y_val)
    history = trainer.train(epochs=50, batch_size=128)
    
    # Visualize training
    plot_training_history(history)
    
    # Initialize predictor
    predictor = HarakatPredictor(model, char_to_idx, diacritic_to_idx, max_len)
    
    # Example prediction
    test_text = "اللغة العربية جميلة"
    print("Input:", test_text)
    print("Output:", predictor.predict(test_text))