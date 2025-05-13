import os
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, 
    Embedding, Dropout, Conv1D, MaxPooling1D,
    Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import unicodedata
from tqdm import tqdm
import glob
import pickle

# Constants
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 50

class ArabicDiacriticsDataProcessor:
    def __init__(self, max_sequence_length=MAX_SEQUENCE_LENGTH):
        self.max_sequence_length = max_sequence_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.diacritic_to_idx = {}
        self.idx_to_diacritic = {}
        
        # Define Arabic diacritics
        self.diacritics = [
            '\u064B',  # Fathatan
            '\u064C',  # Dammatan
            '\u064D',  # Kasratan
            '\u064E',  # Fatha
            '\u064F',  # Damma
            '\u0650',  # Kasra
            '\u0651',  # Shadda
            '\u0652',  # Sukun
            '\u0653',  # Maddah
            '\u0654',  # Hamza above
            '\u0655',  # Hamza below
            '\u0670',  # Superscript Alef
            ''          # No diacritic
        ]
        
        # Initialize diacritic mappings
        for i, diac in enumerate(self.diacritics):
            self.diacritic_to_idx[diac] = i
            self.idx_to_diacritic[i] = diac
    
    def strip_diacritics(self, text):
        """Remove diacritics from Arabic text while preserving characters"""
        # Keep all non-diacritic characters
        return ''.join([c for c in text if c not in self.diacritics])
    
    def extract_diacritics(self, text):
        """Extract diacritics sequence from text"""
        diacritics = []
        for char in text:
            if char in self.diacritics:
                diacritics.append(char)
            else:
                diacritics.append('')  # No diacritic
        return diacritics
    
    def create_character_mappings(self, texts):
        """Create character to index mappings from a list of texts"""
        unique_chars = set()
        for text in texts:
            # Add only non-diacritic characters
            for char in text:
                if char not in self.diacritics:
                    unique_chars.add(char)
        
        # Create mappings
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(unique_chars))}
        self.char_to_idx['<PAD>'] = 0  # Add padding token
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def prepare_dataset(self, diacritized_texts):
        """Prepare dataset from diacritized texts"""
        # Create input (without diacritics) and target (diacritics only) pairs
        inputs = []
        targets = []
        
        for text in diacritized_texts:
            # Remove diacritics for input
            input_text = self.strip_diacritics(text)
            # Extract diacritics for target
            diacritics = []
            
            i = 0
            while i < len(text):
                if i < len(text) and text[i] not in self.diacritics:
                    # Found a character
                    char_diacritics = []
                    j = i + 1
                    # Collect all diacritics following this character
                    while j < len(text) and text[j] in self.diacritics:
                        char_diacritics.append(text[j])
                        j += 1
                    
                    # If no diacritics, add empty string
                    if not char_diacritics:
                        diacritics.append('')
                    else:
                        # Join multiple diacritics if present
                        diacritics.append(''.join(char_diacritics))
                    
                    i = j
                else:
                    i += 1
            
            # Ensure diacritics list matches input text length
            if len(input_text) != len(diacritics):
                print(f"Warning: Mismatch in lengths - text: {len(input_text)}, diacritics: {len(diacritics)}")
                continue
                
            inputs.append(input_text)
            targets.append(diacritics)
        
        # Create character mappings
        self.create_character_mappings(inputs)
        
        return inputs, targets
    
    def encode_input(self, text, pad=True):
        """Encode input text to integer sequence"""
        # Truncate if longer than max length
        if len(text) > self.max_sequence_length:
            text = text[:self.max_sequence_length]
        
        # Convert characters to indices
        encoded = [self.char_to_idx.get(char, 0) for char in text]
        
        # Pad sequence if needed
        if pad and len(encoded) < self.max_sequence_length:
            encoded = encoded + [0] * (self.max_sequence_length - len(encoded))
        
        return encoded
    
    def encode_target(self, diacritics, pad=True):
        """Encode target diacritics to integer sequence"""
        # Truncate if longer than max length
        if len(diacritics) > self.max_sequence_length:
            diacritics = diacritics[:self.max_sequence_length]
        
        # Convert diacritics to indices
        encoded = [self.diacritic_to_idx.get(diac, self.diacritic_to_idx[""]) for diac in diacritics]
        
        # Pad sequence if needed
        if pad and len(encoded) < self.max_sequence_length:
            encoded = encoded + [self.diacritic_to_idx[""]] * (self.max_sequence_length - len(encoded))
        
        return encoded
    
    def decode_diacritics(self, indices):
        """Convert diacritic indices back to diacritics"""
        return [self.idx_to_diacritic[idx] for idx in indices]
    
    def apply_diacritics(self, text, diacritics):
        """Apply diacritics to text"""
        result = ""
        for char, diac in zip(text, diacritics):
            result += char + diac
        return result
    
    def prepare_tf_dataset(self, inputs, targets, batch_size=BATCH_SIZE, shuffle=True):
        """Create TensorFlow dataset from inputs and targets"""
        # Encode inputs and targets
        encoded_inputs = np.array([self.encode_input(text) for text in inputs])
        encoded_targets = np.array([self.encode_target(diac) for diac in targets])
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((encoded_inputs, encoded_targets))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(inputs))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def split_dataset(self, inputs, targets, val_size=0.15, test_size=0.15):
        """Split dataset into train, validation, and test sets"""
        # First split: separate test set
        inputs_train_val, inputs_test, targets_train_val, targets_test = train_test_split(
            inputs, targets, test_size=test_size, random_state=42
        )
        
        # Second split: separate validation set from training set
        val_adjusted_size = val_size / (1 - test_size)
        inputs_train, inputs_val, targets_train, targets_val = train_test_split(
            inputs_train_val, targets_train_val, test_size=val_adjusted_size, random_state=42
        )
        
        return (inputs_train, targets_train), (inputs_val, targets_val), (inputs_test, targets_test)
    
    def data_augmentation(self, inputs, targets, augmentation_factor=0.2):
        """Apply data augmentation specific to Arabic text"""
        augmented_inputs = []
        augmented_targets = []
        
        for input_text, target_diacritics in zip(inputs, targets):
            # Original sample
            augmented_inputs.append(input_text)
            augmented_targets.append(target_diacritics)
            
            # Only augment a portion of the data
            if random.random() > augmentation_factor:
                continue
            
            # 1. Character substitution (similar looking characters)
            if len(input_text) > 5 and random.random() > 0.7:
                char_map = {
                    'ا': 'أ', 'أ': 'ا', 'إ': 'ا',
                    'ه': 'ة', 'ة': 'ه',
                    'ي': 'ى', 'ى': 'ي'
                }
                
                # Choose a random position to substitute
                pos = random.randint(0, len(input_text) - 1)
                if input_text[pos] in char_map:
                    chars = list(input_text)
                    chars[pos] = char_map[input_text[pos]]
                    augmented_inputs.append(''.join(chars))
                    augmented_targets.append(target_diacritics)
            
            # 2. Random character deletion (simulates typos)
            if len(input_text) > 10 and random.random() > 0.8:
                pos = random.randint(0, len(input_text) - 1)
                augmented_input = input_text[:pos] + input_text[pos+1:]
                augmented_target = target_diacritics[:pos] + target_diacritics[pos+1:]
                augmented_inputs.append(augmented_input)
                augmented_targets.append(augmented_target)
        
        return augmented_inputs, augmented_targets


class SimplifiedArabicDiacritizationModel:
    def __init__(
        self, 
        vocab_size, 
        diacritic_size, 
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE
    ):
        self.vocab_size = vocab_size
        self.diacritic_size = diacritic_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self):
        """Build a simplified model for diacritization using BiLSTM and CNN"""
        # Input layer
        inputs = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=False  # Disable masking to avoid broadcasting issues
        )(inputs)
        
        # CNN branch
        conv1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        conv2 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(x)
        conv3 = Conv1D(filters=128, kernel_size=7, padding='same', activation='relu')(x)
        
        # Concatenate CNN outputs
        cnn_features = Concatenate()([conv1, conv2, conv3])
        cnn_features = Dropout(self.dropout_rate)(cnn_features)
        
        # BiLSTM layers
        lstm1 = Bidirectional(LSTM(self.hidden_dim // 2, return_sequences=True))(cnn_features)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        
        lstm2 = Bidirectional(LSTM(self.hidden_dim // 2, return_sequences=True))(lstm1)
        lstm2 = Dropout(self.dropout_rate)(lstm2)
        
        # Output layer
        outputs = Dense(self.diacritic_size, activation='softmax')(lstm2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def compile_model(self, learning_rate=LEARNING_RATE):
        """Compile the model"""
        self.model = self.build_model()
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_callbacks(self, checkpoint_path, patience=5):
        """Get training callbacks"""
        callbacks = [
            # Model checkpointing
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def train(
        self, 
        train_dataset, 
        val_dataset, 
        epochs=EPOCHS, 
        checkpoint_path='model_checkpoints/arabic_diacritization.keras',
        patience=5
    ):
        """Train the model"""
        # Create checkpoint directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Get callbacks
        callbacks = self.get_callbacks(checkpoint_path, patience)
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, test_dataset):
        """Evaluate the model on test data"""
        return self.model.evaluate(test_dataset)
    
    def predict(self, input_sequence):
        """Predict diacritics for input sequence"""
        # Ensure input is in the right format
        if isinstance(input_sequence, str):
            # Convert to batch of one
            input_sequence = np.array([input_sequence])
        
        # Get predictions
        predictions = self.model.predict(input_sequence)
        
        # Convert to diacritic indices
        diacritic_indices = np.argmax(predictions, axis=-1)
        
        return diacritic_indices
    
    def save_model(self, path):
        """Save the model"""
        self.model.save(path)
    
    def load_model(self, path):
        """Load the model"""
        self.model = tf.keras.models.load_model(path)


class DiacritizationEvaluator:
    def __init__(self, data_processor):
        self.data_processor = data_processor
    
    def character_level_accuracy(self, true_diacritics, pred_diacritics):
        """Calculate character-level accuracy"""
        correct = 0
        total = 0
        
        for true_seq, pred_seq in zip(true_diacritics, pred_diacritics):
            for true_diac, pred_diac in zip(true_seq, pred_seq):
                if true_diac == pred_diac:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0
    
    def word_level_accuracy(self, input_texts, true_diacritics, pred_diacritics):
        """Calculate word-level accuracy"""
        correct_words = 0
        total_words = 0
        
        for text, true_seq, pred_seq in zip(input_texts, true_diacritics, pred_diacritics):
            # Split text into words
            words = text.split()
            
            # Track position in the character sequence
            pos = 0
            
            for word in words:
                word_length = len(word)
                
                # Extract diacritics for this word
                true_word_diacritics = true_seq[pos:pos+word_length]
                pred_word_diacritics = pred_seq[pos:pos+word_length]
                
                # Check if all diacritics match
                if true_word_diacritics == pred_word_diacritics:
                    correct_words += 1
                
                total_words += 1
                pos += word_length + 1  # +1 for space
        
        return correct_words / total_words if total_words > 0 else 0
    
    def diacritic_error_rate(self, true_diacritics, pred_diacritics):
        """Calculate diacritic error rate (DER)"""
        errors = 0
        total = 0
        
        for true_seq, pred_seq in zip(true_diacritics, pred_diacritics):
            for true_diac, pred_diac in zip(true_seq, pred_seq):
                if true_diac != pred_diac and true_diac != '':  # Only count errors on actual diacritics
                    errors += 1
                if true_diac != '':
                    total += 1
        
        return errors / total if total > 0 else 0
    
    def evaluate_model(self, model, test_dataset, test_inputs, test_targets):
        """Comprehensive model evaluation"""
        # Get model predictions
        predictions = model.predict(test_dataset)
        pred_indices = np.argmax(predictions, axis=-1)
        
        # Convert indices to diacritics
        true_diacritics = []
        pred_diacritics = []
        
        for i, (input_text, target) in enumerate(zip(test_inputs, test_targets)):
            # Truncate to actual length
            length = min(len(input_text), self.data_processor.max_sequence_length)
            
            # Convert target indices to diacritics
            true_diac = self.data_processor.decode_diacritics(
                self.data_processor.encode_target(target, pad=False)[:length]
            )
            
            # Convert predicted indices to diacritics
            pred_diac = self.data_processor.decode_diacritics(
                pred_indices[i][:length]
            )
            
            true_diacritics.append(true_diac)
            pred_diacritics.append(pred_diac)
        
        # Calculate metrics
        char_accuracy = self.character_level_accuracy(true_diacritics, pred_diacritics)
        word_accuracy = self.word_level_accuracy(test_inputs, true_diacritics, pred_diacritics)
        der = self.diacritic_error_rate(true_diacritics, pred_diacritics)
        
        # Print metrics
        print(f"Character-level accuracy: {char_accuracy:.4f}")
        print(f"Word-level accuracy: {word_accuracy:.4f}")
        print(f"Diacritic error rate: {der:.4f}")
        
        # Show example predictions
        self.show_example_predictions(test_inputs[:5], true_diacritics[:5], pred_diacritics[:5])
        
        return {
            'character_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'diacritic_error_rate': der
        }
    
    def show_example_predictions(self, inputs, true_diacritics, pred_diacritics):
        """Show example predictions"""
        print("\nExample Predictions:")
        print("-" * 80)
        
        for i, (input_text, true_diac, pred_diac) in enumerate(zip(inputs, true_diacritics, pred_diacritics)):
            # Apply diacritics to input text
            true_text = self.data_processor.apply_diacritics(input_text, true_diac)
            pred_text = self.data_processor.apply_diacritics(input_text, pred_diac)
            
            print(f"Example {i+1}:")
            print(f"Input:      {input_text}")
            print(f"True:       {true_text}")
            print(f"Predicted:  {pred_text}")
            print(f"Accuracy:   {sum(t == p for t, p in zip(true_diac, pred_diac)) / len(true_diac):.2f}")
            print("-" * 80)


class ArabicDiacritizer:
    def __init__(self, model_path, processor_path=None):
        self.model = tf.keras.models.load_model(model_path)
        
        if processor_path:
            # Load data processor
            with open(processor_path, 'rb') as f:
                self.processor = pickle.load(f)
        else:
            # Create new processor
            self.processor = ArabicDiacriticsDataProcessor()
    
    def diacritize_text(self, text):
        """Add diacritics to input text"""
        # Preprocess text
        input_text = text.strip()
        
        # Encode input
        encoded_input = np.array([self.processor.encode_input(input_text)])
        
        # Get predictions
        predictions = self.model.predict(encoded_input)
        pred_indices = np.argmax(predictions[0], axis=-1)
        
        # Convert indices to diacritics
        pred_diacritics = self.processor.decode_diacritics(pred_indices[:len(input_text)])
        
        # Apply diacritics to input text
        diacritized_text = self.processor.apply_diacritics(input_text, pred_diacritics)
        
        return diacritized_text
    
    def batch_diacritize(self, texts):
        """Diacritize a batch of texts"""
        results = []
        
        for text in texts:
            results.append(self.diacritize_text(text))
        
        return results


def load_csv_data_from_directory(directory_path, column_name='text_with_harakat'):
    """
    Load all CSV files from a directory and extract a specific column.
    
    Args:
        directory_path (str): Path to the directory containing CSV files
        column_name (str): Name of the column to extract (default: 'text_with_harakat')
    
    Returns:
        list: List of text samples extracted from all CSV files
    """
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory_path}")
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    # Initialize an empty list to store all text samples
    all_samples = []
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if the required column exists
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in {csv_file}. Skipping file.")
                continue
            
            # Extract the text samples and add to the list
            samples = df[column_name].dropna().tolist()
            all_samples.extend(samples)
            
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
    
    print(f"Successfully loaded {len(all_samples)} text samples from {len(csv_files)} CSV files")
    
    return all_samples


def main():
    print("Arabic Diacritization Neural Network (Simplified Version)")
    print("=" * 50)
    
    # 1. Load data from CSV files
    print("Loading data from CSV files...")
    data_directory = "data/"  # Replace with your directory path
    diacritized_texts = load_csv_data_from_directory(data_directory)
    
    # Print some statistics
    print(f"Loaded {len(diacritized_texts)} text samples")
    if diacritized_texts:
        print(f"First sample: {diacritized_texts[0][:100]}...")
    
    # 2. Initialize data processor
    print("Initializing data processor...")
    data_processor = ArabicDiacriticsDataProcessor()
    
    # 3. Prepare dataset
    print("Preparing dataset...")
    inputs, targets = data_processor.prepare_dataset(diacritized_texts)
    
    # Apply data augmentation
    print("Applying data augmentation...")
    augmented_inputs, augmented_targets = data_processor.data_augmentation(inputs, targets)
    
    # 4. Split dataset
    print("Splitting dataset...")
    (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = \
        data_processor.split_dataset(augmented_inputs, augmented_targets)
    
    # 5. Create TensorFlow datasets
    print("Creating TensorFlow datasets...")
    train_dataset = data_processor.prepare_tf_dataset(train_inputs, train_targets)
    val_dataset = data_processor.prepare_tf_dataset(val_inputs, val_targets)
    test_dataset = data_processor.prepare_tf_dataset(test_inputs, test_targets)
    
    # 6. Initialize model
    print("Initializing model...")
    model = SimplifiedArabicDiacritizationModel(
        vocab_size=len(data_processor.char_to_idx),
        diacritic_size=len(data_processor.diacritic_to_idx)
    )
    
    # 7. Compile model
    print("Compiling model...")
    model.compile_model()
    
    # 8. Train model
    print("Training model...")
    os.makedirs('checkpoints', exist_ok=True)
    history = model.train(
        train_dataset,
        val_dataset,
        epochs=50,  # Adjust based on your dataset size
        checkpoint_path='checkpoints/arabic_diacritization_simplified.keras'
    )
    
    # 9. Evaluate model
    print("Evaluating model...")
    evaluator = DiacritizationEvaluator(data_processor)
    metrics = evaluator.evaluate_model(model.model, test_dataset, test_inputs, test_targets)
    
    # 10. Save model and processor
    print("Saving model and processor...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/arabic_diacritization_simplified.keras')
    
    # Save processor for later use
    with open('models/arabic_diacritization_processor.pkl', 'wb') as f:
        pickle.dump(data_processor, f)
    
    print("Training complete!")


if __name__ == "__main__":
    main()