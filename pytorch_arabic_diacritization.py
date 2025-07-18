import os
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import unicodedata
from tqdm import tqdm
import glob
import pickle
import time
import json
from datetime import datetime
import itertools

# Constants
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 128
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
EPOCHS = 500
PATIENCE = 7
IMPROVEMENT_THRESHOLD = 0.00001  # Threshold for early stopping improvement

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

    def clean_text(self, text):
        """Remove punctuation, special chars, digits, and Tatweel (ـ) from text."""
        # Arabic Tatweel char
        tatweel = 'ـ'
        
        # Define all punctuation + special characters to remove (Arabic + ASCII)
        arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
        ascii_punctuations = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        all_punctuations = arabic_punctuations + ascii_punctuations
        
        # Remove Tatweel explicitly
        text = text.replace(tatweel, '')
        
        # Remove punctuation and special characters
        text = ''.join(ch for ch in text if ch not in all_punctuations)
        
        # Remove digits
        text = ''.join(ch for ch in text if not ch.isdigit())
        
        # Optionally normalize spaces
        text = ' '.join(text.split())
        
        return text
    
    def prepare_dataset(self, diacritized_texts):
        """Prepare dataset from diacritized texts"""
        # Create input (without diacritics) and target (diacritics only) pairs
        inputs = []
        targets = []
        
        for text in diacritized_texts:
            # Clean the text before processing
            cleaned_text = self.clean_text(text)
            
            # Remove diacritics for input
            input_text = self.strip_diacritics(cleaned_text)
            
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


# PyTorch Dataset
class ArabicDiacriticsDataset(Dataset):
    def __init__(self, inputs, targets, processor):
        self.inputs = inputs
        self.targets = targets
        self.processor = processor
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_diacritics = self.targets[idx]
        
        # Encode input and target
        encoded_input = self.processor.encode_input(input_text)
        encoded_target = self.processor.encode_target(target_diacritics)
        
        return {
            'input': torch.tensor(encoded_input, dtype=torch.long),
            'target': torch.tensor(encoded_target, dtype=torch.long),
            'length': min(len(input_text), self.processor.max_sequence_length)
        }


# PyTorch Model
class SimplifiedArabicDiacritizationModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        diacritic_size, 
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE
    ):
        super(SimplifiedArabicDiacritizationModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.diacritic_size = diacritic_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # CNN layers
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, 128, kernel_size=7, padding=3)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # BiLSTM layers
        self.lstm1 = nn.LSTM(
            input_size=384,  # 128*3 from concatenated CNN outputs
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, diacritic_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # CNN layers (need to transpose for Conv1d)
        embedded_permuted = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        conv1_out = self.relu(self.conv1(embedded_permuted))  # [batch_size, 128, seq_len]
        conv2_out = self.relu(self.conv2(embedded_permuted))  # [batch_size, 128, seq_len]
        conv3_out = self.relu(self.conv3(embedded_permuted))  # [batch_size, 128, seq_len]
        
        # Concatenate CNN outputs
        cnn_features = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)  # [batch_size, 384, seq_len]
        cnn_features = cnn_features.permute(0, 2, 1)  # [batch_size, seq_len, 384]
        cnn_features = self.dropout(cnn_features)
        
        # BiLSTM layers
        lstm1_out, _ = self.lstm1(cnn_features)  # [batch_size, seq_len, hidden_dim]
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)  # [batch_size, seq_len, hidden_dim]
        lstm2_out = self.dropout(lstm2_out)
        
        # Output layer
        logits = self.fc(lstm2_out)  # [batch_size, seq_len, diacritic_size]
        
        return logits


class DiacritizationTrainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader=None, 
        learning_rate=LEARNING_RATE,
        device=device,
        checkpoint_path=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6
        )
        
        # Load checkpoint if it exists
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Resumed training from epoch {checkpoint['epoch'] + 1}")
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Reshape for loss calculation
            batch_size, seq_len, num_classes = outputs.shape
            outputs_flat = outputs.reshape(-1, num_classes)
            targets_flat = targets.reshape(-1)
            
            # Calculate loss
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Reshape for loss calculation
                batch_size, seq_len, num_classes = outputs.shape
                outputs_flat = outputs.reshape(-1, num_classes)
                targets_flat = targets.reshape(-1)
                
                # Calculate loss
                loss = self.criterion(outputs_flat, targets_flat)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, dim=2)
                mask = (targets != 0)  # Ignore padding
                correct += ((predicted == targets) & mask).sum().item()
                total += mask.sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0
        
        # Update learning rate
        self.scheduler.step(val_loss)
        
        return val_loss, accuracy
    
    def train(self, epochs=EPOCHS, checkpoint_path='checkpoints/pytorch_diacritization.pt', patience=5):
        # Create checkpoint directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            if self.val_loader is not None:
                val_loss, accuracy = self.validate()
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
                
                # Check for improvement
                if val_loss < best_val_loss - IMPROVEMENT_THRESHOLD:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'accuracy': accuracy,
                        'epoch': epoch
                    }, checkpoint_path)
                    print(f"Model saved to {checkpoint_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Train Loss: {train_loss:.4f}")
                
                # Save model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'epoch': epoch
                }, checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
        
        # Load best model
        if self.val_loader is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss {checkpoint['val_loss']:.4f}")
        
        return self.model


class DiacritizationEvaluator:
    def __init__(self, model, processor, device=device):
        self.model = model
        self.processor = processor
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
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
    
    def evaluate(self, test_loader, test_inputs, test_targets):
        """Comprehensive model evaluation"""
        self.model.eval()
        
        # Get model predictions
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move batch to device
                inputs = batch['input'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Get predictions
                _, predictions = torch.max(outputs, dim=2)
                all_predictions.extend(predictions.cpu().numpy())
        
        # Convert indices to diacritics
        true_diacritics = []
        pred_diacritics = []
        
        for i, (input_text, target) in enumerate(zip(test_inputs, test_targets)):
            # Truncate to actual length
            length = min(len(input_text), self.processor.max_sequence_length)
            
            # Convert target indices to diacritics
            true_diac = self.processor.decode_diacritics(
                self.processor.encode_target(target, pad=False)[:length]
            )
            
            # Convert predicted indices to diacritics
            pred_diac = self.processor.decode_diacritics(
                all_predictions[i][:length]
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
            true_text = self.processor.apply_diacritics(input_text, true_diac)
            pred_text = self.processor.apply_diacritics(input_text, pred_diac)
            
            print(f"Example {i+1}:")
            print(f"Input:      {input_text}")
            print(f"True:       {true_text}")
            print(f"Predicted:  {pred_text}")
            print(f"Accuracy:   {sum(t == p for t, p in zip(true_diac, pred_diac)) / len(true_diac):.2f}")
            print("-" * 80)


class ArabicDiacritizer:
    def __init__(self, model_path, processor_path=None, device=device):
        self.device = device
        
        # Load processor
        if processor_path:
            with open(processor_path, 'rb') as f:
                self.processor = pickle.load(f)
        else:
            self.processor = ArabicDiacriticsDataProcessor()
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        self.model = SimplifiedArabicDiacritizationModel(
            vocab_size=len(self.processor.char_to_idx),
            diacritic_size=len(self.processor.diacritic_to_idx)
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def diacritize_text(self, text):
        """Add diacritics to input text"""
        # Preprocess text
        input_text = text.strip()
        
        # Encode input
        encoded_input = self.processor.encode_input(input_text)
        input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, pred_indices = torch.max(outputs, dim=2)
        
        # Convert indices to diacritics
        pred_diacritics = self.processor.decode_diacritics(pred_indices[0].cpu().numpy()[:len(input_text)])
        
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


def train_model(
    max_sequence_length=256, 
    batch_size=32, 
    embedding_dim=256, 
    hidden_dim=512, 
    dropout_rate=0.2, 
    learning_rate=1e-3, 
    epochs=50,
    patience=5,
    data_directory="data/",
    results_dir="results/"
):
    """Train a model with the given hyperparameters and return metrics"""
    
    # Create model name based on hyperparameters
    model_name = f"ADNN_{max_sequence_length}_{batch_size}_{embedding_dim}_{hidden_dim}_{dropout_rate}_{learning_rate}_{epochs}"
    print(f"\n{'='*80}\nTraining model: {model_name}\n{'='*80}")
    
    # Create directories
    os.makedirs(f"checkpoints/{model_name}", exist_ok=True)
    os.makedirs(f"models/{model_name}", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    checkpoint_path = f"checkpoints/{model_name}/{model_name}.pt"
    processor_path = f"models/{model_name}/{model_name}_processor.pkl"
    model_path = f"models/{model_name}/{model_name}.pt"
    
    # 1. Load data from CSV files
    print("Loading data from CSV files...")
    diacritized_texts = load_csv_data_from_directory(data_directory)
    
    # 2. Initialize data processor
    print("Initializing data processor...")
    data_processor = ArabicDiacriticsDataProcessor(max_sequence_length=max_sequence_length)
    
    # 3. Prepare dataset
    print("Preparing dataset...")
    inputs, targets = data_processor.prepare_dataset(diacritized_texts)
    
    # Apply data augmentation
    print("Applying data augmentation...")
    augmented_inputs, augmented_targets = data_processor.data_augmentation(inputs, targets)
    
    # 4. Split dataset
    print("Splitting dataset...")
    # First split: separate test set
    inputs_train_val, inputs_test, targets_train_val, targets_test = train_test_split(
        augmented_inputs, augmented_targets, test_size=0.15, random_state=42
    )
    
    # Second split: separate validation set from training set
    val_adjusted_size = 0.15 / (1 - 0.15)  # Adjust validation size
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        inputs_train_val, targets_train_val, test_size=val_adjusted_size, random_state=42
    )
    
    print(f"Train set: {len(inputs_train)} samples")
    print(f"Validation set: {len(inputs_val)} samples")
    print(f"Test set: {len(inputs_test)} samples")
    
    # 5. Create PyTorch datasets and dataloaders
    print("Creating PyTorch datasets and dataloaders...")
    train_dataset = ArabicDiacriticsDataset(inputs_train, targets_train, data_processor)
    val_dataset = ArabicDiacriticsDataset(inputs_val, targets_val, data_processor)
    test_dataset = ArabicDiacriticsDataset(inputs_test, targets_test, data_processor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 6. Initialize model
    print("Initializing model...")
    model = SimplifiedArabicDiacritizationModel(
        vocab_size=len(data_processor.char_to_idx),
        diacritic_size=len(data_processor.diacritic_to_idx),
        max_sequence_length=max_sequence_length,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 7. Train model
    print("Training model...")
    start_time = time.time()
    
    trainer = DiacritizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        device=device,
        checkpoint_path=checkpoint_path
    )
    
    model = trainer.train(
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        patience=patience
    )
    
    training_time = time.time() - start_time
    
    # 8. Evaluate model
    print("Evaluating model...")
    evaluator = DiacritizationEvaluator(model, data_processor, device)
    
    # Measure inference time
    inference_start_time = time.time()
    metrics = evaluator.evaluate(test_loader, inputs_test, targets_test)
    inference_time = time.time() - inference_start_time
    
    # 9. Save model and processor
    print("Saving model and processor...")
    
    # Save processor for later use
    with open(processor_path, 'wb') as f:
        pickle.dump(data_processor, f)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'max_sequence_length': max_sequence_length,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'dropout_rate': dropout_rate
        }
    }, model_path)
    
    # 10. Save results
    results = {
        'model_name': model_name,
        'hyperparameters': {
            'max_sequence_length': max_sequence_length,
            'batch_size': batch_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'patience': patience
        },
        'metrics': metrics,
        'model_size': total_params,
        'training_time': training_time,
        'inference_time': inference_time,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results to JSON
    with open(f"{results_dir}/{model_name}_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Training complete for {model_name}!")
    print(f"Results saved to {results_dir}/{model_name}_results.json")
    
    return results


def run_hyperparameter_optimization():
    """Run hyperparameter optimization with multiple combinations"""
    
    # Define hyperparameter grid
    param_grid = {
        'max_sequence_length': [512],
        'batch_size': [128],
        'embedding_dim': [256],
        'hidden_dim': [512],
        'dropout_rate': [0.3],
        'learning_rate': [1e-3],
        'epochs': [500],  # Set to a high value, early stopping will prevent overfitting
        'patience': [7]
    }
    
    # Create all combinations of hyperparameters
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[key] for key in keys]))
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/optimization_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save parameter grid for reference
    with open(f"{results_dir}/param_grid.json", 'w') as f:
        json.dump(param_grid, f, indent=4)
    
    # Train models with each combination
    all_results = []
    
    print(f"Starting hyperparameter optimization with {len(combinations)} combinations")
    
    for i, combination in enumerate(combinations):
        print(f"\nCombination {i+1}/{len(combinations)}")
        
        # Create parameter dictionary
        params = dict(zip(keys, combination))
        
        # Train model with these parameters
        try:
            result = train_model(**params, results_dir=results_dir)
            all_results.append(result)
        except Exception as e:
            print(f"Error training model with parameters {params}: {str(e)}")
            # Log the error
            with open(f"{results_dir}/errors.log", 'a') as f:
                f.write(f"Error with parameters {params}: {str(e)}\n")
    
    # Save all results
    with open(f"{results_dir}/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    
    return all_results


def create_comparison_report(results, results_dir):
    """Create a comparison report of all models"""
    
    if not results:
        print("No results to compare")
        return
    
    # Extract key metrics for comparison
    comparison = []
    
    for result in results:
        comparison.append({
            'model_name': result['model_name'],
            'character_accuracy': result['metrics']['character_accuracy'],
            'word_accuracy': result['metrics']['word_accuracy'],
            'diacritic_error_rate': result['metrics']['diacritic_error_rate'],
            'model_size': result['model_size'],
            'training_time': result['training_time'],
            'inference_time': result['inference_time'],
            **result['hyperparameters']
        })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(comparison)
    
    # Sort by character accuracy (descending)
    df_sorted = df.sort_values('character_accuracy', ascending=False)
    
    # Save comparison to CSV
    df_sorted.to_csv(f"{results_dir}/comparison.csv", index=False)
    
    # Create visualizations
    create_visualizations(df, results_dir)
    
    # Print top 3 models
    print("\nTop 3 models by character accuracy:")
    for i, row in df_sorted.head(3).iterrows():
        print(f"{i+1}. {row['model_name']}: {row['character_accuracy']:.4f}")
    
    return df_sorted


def create_visualizations(df, results_dir):
    """Create visualizations for hyperparameter comparison"""
    
    # Create directory for visualizations
    viz_dir = f"{results_dir}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    
    # 1. Accuracy vs Model Size
    plt.figure(figsize=(10, 6))
    plt.scatter(df['model_size'], df['character_accuracy'], alpha=0.7)
    for i, row in df.iterrows():
        plt.annotate(row['model_name'].split('_')[1:4], 
                    (row['model_size'], row['character_accuracy']),
                    fontsize=8)
    plt.xlabel('Model Size (parameters)')
    plt.ylabel('Character Accuracy')
    plt.title('Model Size vs Accuracy')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/model_size_vs_accuracy.png", dpi=300)
    plt.close()
    
    # 2. Training Time vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(df['training_time'] / 60, df['character_accuracy'], alpha=0.7)
    plt.xlabel('Training Time (minutes)')
    plt.ylabel('Character Accuracy')
    plt.title('Training Time vs Accuracy')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/training_time_vs_accuracy.png", dpi=300)
    plt.close()
    
    # 3. Hyperparameter Heatmaps - MODIFIED THIS SECTION
    # First check if we have enough data for heatmaps
    if len(df) >= 4:  # Need at least a few data points for meaningful heatmaps
        for param in ['max_sequence_length', 'batch_size', 'embedding_dim', 'hidden_dim', 'learning_rate']:
            # Skip pivot tables and use simpler visualizations
            if len(df[param].unique()) > 1:
                plt.figure(figsize=(10, 6))
                # Group by parameter and calculate mean accuracy
                grouped = df.groupby(param)['character_accuracy'].mean().reset_index()
                plt.bar(grouped[param].astype(str), grouped['character_accuracy'])
                plt.xlabel(param)
                plt.ylabel('Mean Character Accuracy')
                plt.title(f'Impact of {param} on Accuracy')
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/{param}_impact.png", dpi=300)
                plt.close()
        
        # Special handling for dropout_rate
        if 'dropout_rate' in df.columns and len(df['dropout_rate'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            # Ensure dropout_rate is treated as a simple value
            df['dropout_rate_str'] = df['dropout_rate'].astype(str)
            grouped = df.groupby('dropout_rate_str')['character_accuracy'].mean().reset_index()
            plt.bar(grouped['dropout_rate_str'], grouped['character_accuracy'])
            plt.xlabel('Dropout Rate')
            plt.ylabel('Mean Character Accuracy')
            plt.title('Impact of Dropout Rate on Accuracy')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/dropout_rate_impact.png", dpi=300)
            plt.close()
    
    # 4. Bar chart of top models
    top_models = df.sort_values('character_accuracy', ascending=False).head(5)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_models['model_name'], top_models['character_accuracy'])
    plt.xlabel('Model')
    plt.ylabel('Character Accuracy')
    plt.title('Top 5 Models by Character Accuracy')
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/top_models.png", dpi=300)
    plt.close()
    
    # 5. Correlation matrix - MODIFIED THIS SECTION
    try:
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove any problematic columns
        safe_cols = [col for col in numeric_cols if df[col].apply(lambda x: isinstance(x, (int, float))).all()]
        
        if len(safe_cols) >= 2:  # Need at least 2 columns for correlation
            corr = df[safe_cols].corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Matrix of Hyperparameters and Metrics')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/correlation_matrix.png", dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error creating correlation matrix: {str(e)}")
    
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Hyperparameters and Metrics')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/correlation_matrix.png", dpi=300)
    plt.close()


def main():
    """Main function to run hyperparameter optimization"""
    print("Arabic Diacritization Neural Network - Hyperparameter Optimization")
    print("=" * 80)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Run hyperparameter optimization
    results = run_hyperparameter_optimization()
    
    print("\nHyperparameter optimization complete!")
    print(f"Results saved to results/optimization_*")


if __name__ == "__main__":
    main()
